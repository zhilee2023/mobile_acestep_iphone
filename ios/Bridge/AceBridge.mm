#import "AceBridge.h"
#import "FixedSongEngine.h"
#import "EncoderBenchmark.h"
#import <MNN/Interpreter.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

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
                             @"vae_decoder_174_w4.mnn"]) {
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

+ (NSDictionary<NSString *, id> *)diagnoseVAEAtDirectory:(NSString *)directory {
    return [FixedSongEngine diagnoseVAEAtDirectory:directory];
}

@end

