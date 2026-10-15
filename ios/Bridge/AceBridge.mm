#import "AceBridge.h"
#import "FixedSongEngine.h"
#import "EncoderBenchmark.h"
#import "LyricConditionEngine.h"
#import <MNN/Interpreter.hpp>
#import <os/proc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <unistd.h>

namespace {
NSDictionary<NSString *, id> *failure(NSString *message) {
    return @{ @"success": @NO, @"message": message };
}

NSData *probeData(NSString *name) {
    NSURL *url = [NSBundle.mainBundle URLForResource:name withExtension:@"f32" subdirectory:@"Probe"];
    return url ? [NSData dataWithContentsOfURL:url] : nil;
}
}

@implementation AceBridge

+ (NSDictionary<NSString *, NSNumber *> *)modelStatusAtDirectory:(NSString *)directory {
    NSFileManager *manager = NSFileManager.defaultManager;
    NSMutableDictionary<NSString *, NSNumber *> *status = [NSMutableDictionary dictionary];
    for (NSString *name in @[@"text_encoder_w4.mnn", @"text_encoder_w4.mnn.weight",
                             @"dit_w4.mnn", @"dit_w4.mnn.weight",
                             @"vae_decoder_192_w4.mnn", @"vae_decoder_256_w4.mnn",
                             @"vae_decoder_174_w4.mnn", @"lyric_embedding.f16",
                             @"lyric_encoder_73_w4.mnn", @"text_projector.f16"]) {
        NSString *path = [directory stringByAppendingPathComponent:name];
        NSDictionary *attributes = [manager attributesOfItemAtPath:path error:nil];
        status[name] = @([attributes[NSFileSize] unsignedLongLongValue] > 0);
    }
    return status;
}

+ (NSDictionary<NSString *, id> *)runDiTProbeAtDirectory:(NSString *)directory {
    @autoreleasepool {
        NSString *modelPath = [directory stringByAppendingPathComponent:@"dit_w4.mnn"];
        NSString *weightPath = [directory stringByAppendingPathComponent:@"dit_w4.mnn.weight"];
        if (![NSFileManager.defaultManager fileExistsAtPath:modelPath] ||
            ![NSFileManager.defaultManager fileExistsAtPath:weightPath]) {
            return failure(@"DiT 模型或外部权重缺失。");
        }

        auto net = std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)>(
            MNN::Interpreter::createFromFile(modelPath.fileSystemRepresentation),
            &MNN::Interpreter::destroy);
        if (!net) return failure(@"无法加载 DiT MNN 模型。");

        MNN::BackendConfig backend;
        backend.precision = MNN::BackendConfig::Precision_High;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_METAL;
        config.backupType = MNN_FORWARD_CPU;
        config.backendConfig = &backend;
        MNN::Session *session = net->createSession(config);
        if (!session) return failure(@"无法创建 MNN Metal 会话；可能是内存不足或算子不支持。");

        NSDictionary<NSString *, NSString *> *inputs = @{
            @"hidden_states": @"dit-hidden", @"timestep": @"dit-time",
            @"timestep_r": @"dit-time-r", @"encoder_hidden_states": @"dit-encoder",
            @"context_latents": @"dit-context"
        };
        for (NSString *name in inputs) {
            NSData *data = probeData(inputs[name]);
            MNN::Tensor *input = net->getSessionInput(session, name.UTF8String);
            if (!input || !data) {
                net->releaseSession(session);
                return failure([NSString stringWithFormat:@"DiT 测试输入 %@ 缺失或形状不符。", name]);
            }
            MNN::Tensor host(input, input->getDimensionType(), true);
            if (!host.host<float>() || data.length != host.usize()) {
                net->releaseSession(session);
                return failure([NSString stringWithFormat:@"DiT 测试输入 %@ 缺失或形状不符。", name]);
            }
            std::memcpy(host.host<float>(), data.bytes, data.length);
            if (!input->copyFromHostTensor(&host)) {
                if (!input->host<float>()) {
                    net->releaseSession(session);
                    return failure([NSString stringWithFormat:@"无法传入 %@。", name]);
                }
                std::memcpy(input->host<float>(), data.bytes, data.length);
            }
        }

        CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        MNN::ErrorCode code = net->runSession(session);
        double seconds = CFAbsoluteTimeGetCurrent() - started;
        if (code != MNN::NO_ERROR) {
            net->releaseSession(session);
            return failure([NSString stringWithFormat:@"MNN DiT 单步运行失败，错误码 %d。", static_cast<int>(code)]);
        }

        MNN::Tensor *output = net->getSessionOutput(session, "denoised");
        NSData *golden = probeData(@"dit-golden");
        if (!output || !golden) {
            net->releaseSession(session);
            return failure(@"DiT 输出形状与参考结果不符。");
        }
        MNN::Tensor host(output, output->getDimensionType(), true);
        if (!host.host<float>() || host.usize() != golden.length) {
            net->releaseSession(session);
            return failure(@"DiT 输出形状与参考结果不符。");
        }
        if (!output->copyToHostTensor(&host)) {
            if (!output->host<float>()) {
                net->releaseSession(session);
                return failure(@"无法读取 DiT 输出。");
            }
            std::memcpy(host.host<float>(), output->host<float>(), golden.length);
        }
        const float *actual = host.host<float>();
        const float *expected = static_cast<const float *>(golden.bytes);
        double squaredError = 0.0;
        double squaredReference = 0.0;
        for (size_t index = 0; index < golden.length / sizeof(float); ++index) {
            if (!std::isfinite(actual[index])) {
                net->releaseSession(session);
                return failure(@"DiT 输出包含非有限数值。");
            }
            double difference = static_cast<double>(actual[index]) - expected[index];
            squaredError += difference * difference;
            squaredReference += static_cast<double>(expected[index]) * expected[index];
        }
        double relativeL2 = std::sqrt(squaredError / std::max(squaredReference, 1e-20));
        net->releaseSession(session);
        return @{ @"success": @YES, @"message": @"DiT W4 已在 iPhone 运行一次。Metal 已请求；未排除 CPU 算子回退。",
                  @"elapsedSeconds": @(seconds), @"relativeL2": @(relativeL2) };
    }
}

+ (NSDictionary<NSString *, id> *)generateFixedSongAtDirectory:(NSString *)directory {
    return [FixedSongEngine generateAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)runTwoStepHighProbeAtDirectory:(NSString *)directory {
    return [FixedSongEngine probeTwoStepHighAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)runTwoStepLowProbeAtDirectory:(NSString *)directory {
    return [FixedSongEngine probeTwoStepLowAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)runEncoderBenchmarkAtDirectory:(NSString *)directory {
    return [EncoderBenchmark runAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)runEncoderDiTSerialProbeAtDirectory:(NSString *)directory {
    return [EncoderBenchmark runSerialDiTAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)generateFullSerialSongAtDirectory:(NSString *)directory {
    return [EncoderBenchmark runFullSerialSongAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)generateOptimizedSerialSongAtDirectory:(NSString *)directory {
    return [EncoderBenchmark runOptimizedFullSerialAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)generateCpuVaeSerialSongAtDirectory:(NSString *)directory {
    return [EncoderBenchmark runOptimizedCpuVaeSerialAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)generateNoTuneVAESongAtDirectory:(NSString *)directory {
    return [FixedSongEngine generateOptimizedNoTuneVAEAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)diagnoseVAEAtDirectory:(NSString *)directory {
    return [FixedSongEngine diagnoseVAEAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)diagnoseVAENoTuneAtDirectory:(NSString *)directory {
    return [FixedSongEngine diagnoseVAENoTuneAtDirectory:directory];
}

+ (NSDictionary<NSString *, id> *)generateCustomLyrics:(NSString *)lyrics atDirectory:(NSString *)directory {
    const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    NSDictionary<NSString *, id> *encoded = [LyricConditionEngine encodeLyrics:lyrics atDirectory:directory];
    if (![encoded[@"success"] boolValue]) return encoded;
    NSData *condition = encoded[@"condition"];
    // The lyric graph has left scope; wait until its Metal allocations are
    // released before constructing the approximately 2 GB DiT session.
    int consecutive = 0;
    const CFAbsoluteTime waitStarted = CFAbsoluteTimeGetCurrent();
    while (CFAbsoluteTimeGetCurrent() - waitStarted < 8.0) {
        consecutive = os_proc_available_memory() >= 2400000000ULL ? consecutive + 1 : 0;
        if (consecutive >= 2) break;
        usleep(250000);
    }
    if (consecutive < 2) return failure(@"歌词已编码，但内存尚未回收到安全范围；未启动 DiT。请稍后重试。");
    NSDictionary<NSString *, id> *song = [FixedSongEngine generateWithCondition:condition atDirectory:directory];
    if (![song[@"success"] boolValue]) return song;
    const double elapsed = CFAbsoluteTimeGetCurrent() - started;
    return @{ @"success": @YES, @"wavPath": song[@"wavPath"],
              @"elapsedSeconds": @(elapsed), @"audioSeconds": @30.0,
              @"backend": @"端侧歌词编码 → DiT → VAE / Metal 请求、CPU 回退可能",
              @"message": [NSString stringWithFormat:@"新歌词已在手机编码并接入 DiT；30 秒音频生成成功，耗时 %.1f 秒，RTF %.2f。风格仍为固定样例。", elapsed, elapsed / 30.0] };
}

+ (NSDictionary<NSString *, id> *)generateCustomSongWithLyrics:(NSString *)lyrics
                                                     caption:(NSString *)caption
                                                 atDirectory:(NSString *)directory {
    const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    NSDictionary<NSString *, id> *tokens = [LyricConditionEngine tokenizeCaption:caption];
    if (![tokens[@"success"] boolValue]) return tokens;
    NSDictionary<NSString *, id> *style = [EncoderBenchmark encodeStyleIDs:tokens[@"ids"] atDirectory:directory];
    if (![style[@"success"] boolValue]) return style;
    NSData *styleData = style[@"style"];
    if (styleData.length != 107ULL * 2048 * sizeof(float)) return failure(@"Qwen 风格投影形状不正确。");
    const double styleSeconds = CFAbsoluteTimeGetCurrent() - started;

    // Never hold Qwen, lyric encoder, and DiT sessions at the same time.
    int consecutive = 0;
    const CFAbsoluteTime firstWait = CFAbsoluteTimeGetCurrent();
    while (CFAbsoluteTimeGetCurrent() - firstWait < 10.0) {
        consecutive = os_proc_available_memory() >= 2400000000ULL ? consecutive + 1 : 0;
        if (consecutive >= 2) break;
        usleep(250000);
    }
    if (consecutive < 2) return failure(@"Qwen 已完成，但内存尚未回收到 2400 MB；未启动歌词编码，请稍后重试。");
    const CFAbsoluteTime lyricStarted = CFAbsoluteTimeGetCurrent();
    NSDictionary<NSString *, id> *lyric = [LyricConditionEngine encodeLyrics:lyrics atDirectory:directory];
    if (![lyric[@"success"] boolValue]) return lyric;
    NSData *lyricCondition = lyric[@"condition"];
    if (lyricCondition.length != 181ULL * 2048 * sizeof(float)) return failure(@"歌词编码条件形状不正确。");
    NSMutableData *condition = [lyricCondition mutableCopy];
    std::memcpy(static_cast<uint8_t *>(condition.mutableBytes) + 74ULL * 2048 * sizeof(float),
                styleData.bytes, styleData.length);
    const double lyricSeconds = CFAbsoluteTimeGetCurrent() - lyricStarted;

    consecutive = 0;
    const CFAbsoluteTime secondWait = CFAbsoluteTimeGetCurrent();
    while (CFAbsoluteTimeGetCurrent() - secondWait < 10.0) {
        consecutive = os_proc_available_memory() >= 2400000000ULL ? consecutive + 1 : 0;
        if (consecutive >= 2) break;
        usleep(250000);
    }
    if (consecutive < 2) return failure(@"歌词已完成，但内存尚未回收到 2400 MB；未启动 DiT，请稍后重试。");
    NSDictionary<NSString *, id> *song = [FixedSongEngine generateWithConditionCpuVAE:condition atDirectory:directory];
    if (![song[@"success"] boolValue]) return song;
    const double elapsed = CFAbsoluteTimeGetCurrent() - started;
    return @{ @"success": @YES, @"wavPath": song[@"wavPath"],
              @"elapsedSeconds": @(elapsed), @"audioSeconds": @30.0,
              @"backend": @"Qwen 风格 + ACE 歌词编码 → DiT Metal High → VAE CPU Low",
              @"message": [NSString stringWithFormat:@"手机已接通风格、歌词、DiT 和 CPU VAE；30 秒音频耗时 %.1f 秒，RTF %.2f。风格编码 %.1f 秒，歌词编码 %.1f 秒。",
                           elapsed, elapsed / 30.0, styleSeconds, lyricSeconds] };
}

@end

