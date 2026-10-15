#import "FixedSongEngine.h"
#import <MNN/Interpreter.hpp>
#import <mach/mach.h>
#import <os/proc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <unistd.h>

namespace {
constexpr int kLatentFrames = 750;
constexpr int kLatentChannels = 64;
constexpr int kSampleRate = 48000;
constexpr int kUpsample = 1920;
constexpr int kSamples = kLatentFrames * kUpsample;
constexpr int kVaeCoreFrames = 64;
constexpr int kVaeOverlapFrames = 64;
constexpr int kVaeWindowFrames = kVaeCoreFrames + 2 * kVaeOverlapFrames;
constexpr int kVaeChunks = (kLatentFrames + kVaeCoreFrames - 1) / kVaeCoreFrames;
constexpr size_t kMinimumAvailableBytes = 512ULL * 1024 * 1024;
constexpr std::array<float, 8> kTimesteps = {1.0f, 0.875f, 0.75f, 0.625f,
                                              0.5f, 0.375f, 0.25f, 0.125f};

NSDictionary<NSString *, id> *failure(NSString *message) {
    return @{ @"success": @NO, @"message": message };
}

struct InferenceTrace {
    FILE *file = nullptr;

    InferenceTrace() {
        NSURL *documents = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory
                                                                 inDomains:NSUserDomainMask].firstObject;
        NSURL *directory = [documents URLByAppendingPathComponent:@"Generated" isDirectory:YES];
        if (![NSFileManager.defaultManager createDirectoryAtURL:directory
                                    withIntermediateDirectories:YES attributes:nil error:nil]) return;
        NSURL *url = [directory URLByAppendingPathComponent:@"last-inference.log"];
        file = std::fopen(url.fileSystemRepresentation, "w");
        record(@"开始：30 秒固定样例；Metal 低精度／低内存模式");
    }

    ~InferenceTrace() {
        if (file) std::fclose(file);
    }

    size_t availableBytes() const { return os_proc_available_memory(); }

    void record(NSString *phase) {
        if (!file) return;
        task_vm_info_data_t info = {};
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        uint64_t footprint = 0;
        if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count)
                == KERN_SUCCESS) footprint = info.phys_footprint;
        std::fprintf(file, "%.3f\t%s\tfootprint=%llu MB\tavailable=%llu MB\n",
                     CFAbsoluteTimeGetCurrent(), phase.UTF8String,
                     static_cast<unsigned long long>(footprint / 1000000),
                     static_cast<unsigned long long>(availableBytes() / 1000000));
        std::fflush(file);
        fsync(fileno(file));
    }

    bool hasHeadroom() const { return availableBytes() >= kMinimumAvailableBytes; }
};

struct Graph {
    MNN::Interpreter *interpreter = nullptr;
    MNN::Session *session = nullptr;
    ~Graph() {
        if (interpreter && session) interpreter->releaseSession(session);
        if (interpreter) MNN::Interpreter::destroy(interpreter);
    }
};

struct TensorStats {
    float peak = 0.0f;
    size_t invalidIndex = 0;
    bool finite = true;
};

TensorStats tensorStats(const std::vector<float> &values) {
    TensorStats result;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            result.invalidIndex = index;
            result.finite = false;
            return result;
        }
        result.peak = std::max(result.peak, std::abs(values[index]));
    }
    return result;
}

std::unique_ptr<Graph> openGraph(NSString *path, InferenceTrace &trace) {
    auto graph = std::make_unique<Graph>();
    const CFAbsoluteTime loadStarted = CFAbsoluteTimeGetCurrent();
    graph->interpreter = MNN::Interpreter::createFromFile(path.fileSystemRepresentation);
    trace.record([NSString stringWithFormat:@"%@：读取模型 %.3f 秒", path.lastPathComponent,
                  CFAbsoluteTimeGetCurrent() - loadStarted]);
    if (!graph->interpreter) return nullptr;
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_Low;
    backend.memory = MNN::BackendConfig::Memory_Low;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_METAL;
    config.backupType = MNN_FORWARD_CPU;
    config.backendConfig = &backend;
    const CFAbsoluteTime sessionStarted = CFAbsoluteTimeGetCurrent();
    graph->session = graph->interpreter->createSession(config);
    trace.record([NSString stringWithFormat:@"%@：创建会话 %.3f 秒", path.lastPathComponent,
                  CFAbsoluteTimeGetCurrent() - sessionStarted]);
    if (!graph->session) return nullptr;
    const CFAbsoluteTime releaseStarted = CFAbsoluteTimeGetCurrent();
    graph->interpreter->releaseModel();
    trace.record([NSString stringWithFormat:@"%@：释放原始模型 %.3f 秒", path.lastPathComponent,
                  CFAbsoluteTimeGetCurrent() - releaseStarted]);
    return graph;
}

bool setInput(Graph &graph, const char *name, const float *values, size_t count) {
    MNN::Tensor *input = graph.interpreter->getSessionInput(graph.session, name);
    if (!input) return false;
    MNN::Tensor host(input, input->getDimensionType(), true);
    if (!host.host<float>() || host.usize() != count * sizeof(float)) return false;
    std::memcpy(host.host<float>(), values, count * sizeof(float));
    if (!input->copyFromHostTensor(&host)) {
        if (!input->host<float>()) return false;
        std::memcpy(input->host<float>(), values, count * sizeof(float));
    }
    return true;
}

bool readOutput(Graph &graph, const char *name, size_t count, std::vector<float> &values) {
    MNN::Tensor *output = graph.interpreter->getSessionOutput(graph.session, name);
    if (!output) return false;
    MNN::Tensor host(output, output->getDimensionType(), true);
    if (!host.host<float>() || host.usize() != count * sizeof(float)) return false;
    if (!output->copyToHostTensor(&host)) {
        if (!output->host<float>()) return false;
        std::memcpy(host.host<float>(), output->host<float>(), count * sizeof(float));
    }
    values.assign(host.host<float>(), host.host<float>() + count);
    return true;
}

bool loadFixture(NSString *name, size_t count, std::vector<float> &values) {
    NSURL *url = [NSBundle.mainBundle URLForResource:name withExtension:@"f32" subdirectory:@"Probe"];
    NSData *data = url ? [NSData dataWithContentsOfURL:url] : nil;
    if (!data || data.length != count * sizeof(float)) return false;
    values.resize(count);
    std::memcpy(values.data(), data.bytes, data.length);
    return true;
}

void put16(std::vector<uint8_t> &data, size_t offset, uint16_t value) {
    data[offset] = static_cast<uint8_t>(value);
    data[offset + 1] = static_cast<uint8_t>(value >> 8);
}

void put32(std::vector<uint8_t> &data, size_t offset, uint32_t value) {
    for (int index = 0; index < 4; ++index) data[offset + index] = static_cast<uint8_t>(value >> (8 * index));
}

NSURL *saveWAV(const std::vector<float> &audio, NSString **reason) {
    float peak = 0.0f;
    for (size_t index = 0; index < audio.size(); ++index) {
        float sample = audio[index];
        if (!std::isfinite(sample)) {
            if (reason) *reason = [NSString stringWithFormat:@"VAE 输出含非有限值，样本位置 %zu。", index];
            return nil;
        }
        peak = std::max(peak, std::abs(sample));
    }
    if (peak < 1e-6f) {
        if (reason) *reason = [NSString stringWithFormat:@"VAE 输出接近静音，峰值 %.8g。", peak];
        return nil;
    }
    const float gain = std::pow(10.0f, -1.0f / 20.0f) / peak;
    constexpr size_t bytes = static_cast<size_t>(kSamples) * 2 * sizeof(int16_t);
    std::vector<uint8_t> wave(44 + bytes);
    std::memcpy(wave.data(), "RIFF", 4);
    put32(wave, 4, static_cast<uint32_t>(36 + bytes));
    std::memcpy(wave.data() + 8, "WAVEfmt ", 8);
    put32(wave, 16, 16);
    put16(wave, 20, 1);
    put16(wave, 22, 2);
    put32(wave, 24, kSampleRate);
    put32(wave, 28, kSampleRate * 2 * sizeof(int16_t));
    put16(wave, 32, 2 * sizeof(int16_t));
    put16(wave, 34, 16);
    std::memcpy(wave.data() + 36, "data", 4);
    put32(wave, 40, static_cast<uint32_t>(bytes));
    for (int sample = 0; sample < kSamples; ++sample) {
        for (int channel = 0; channel < 2; ++channel) {
            float scaled = std::clamp(audio[channel * kSamples + sample] * gain, -1.0f, 1.0f);
            int16_t pcm = static_cast<int16_t>(std::lrintf(scaled * 32767.0f));
            put16(wave, 44 + (static_cast<size_t>(sample) * 2 + channel) * sizeof(int16_t),
                  static_cast<uint16_t>(pcm));
        }
    }
    NSURL *documents = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
    NSURL *directory = [documents URLByAppendingPathComponent:@"Generated" isDirectory:YES];
    NSError *directoryError = nil;
    if (![NSFileManager.defaultManager createDirectoryAtURL:directory
                                withIntermediateDirectories:YES attributes:nil error:&directoryError]) {
        if (reason) *reason = [NSString stringWithFormat:@"创建音频目录失败：%@", directoryError.localizedDescription];
        return nil;
    }
    NSString *filename = [NSString stringWithFormat:@"fixed-30s-w4-%@.wav", [[NSUUID UUID] UUIDString]];
    NSURL *target = [directory URLByAppendingPathComponent:filename];
    NSData *data = [NSData dataWithBytesNoCopy:wave.data() length:wave.size() freeWhenDone:NO];
    NSError *writeError = nil;
    if (![data writeToURL:target options:NSDataWritingAtomic error:&writeError]) {
        if (reason) *reason = [NSString stringWithFormat:@"写入 WAV 失败：%@", writeError.localizedDescription];
        return nil;
    }
    return target;
}
} // namespace

@implementation FixedSongEngine

+ (NSDictionary<NSString *, id> *)generateAtDirectory:(NSString *)directory {
    @autoreleasepool {
        InferenceTrace trace;
        const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        const size_t latentCount = static_cast<size_t>(kLatentFrames) * kLatentChannels;
        std::vector<float> hidden, encoder, context;
        if (!loadFixture(@"dit-hidden", latentCount, hidden) ||
            !loadFixture(@"dit-encoder", 181 * 2048, encoder) ||
            !loadFixture(@"dit-context", static_cast<size_t>(kLatentFrames) * 128, context)) {
            trace.record(@"失败：固定条件缺失");
            return failure(@"固定歌曲条件文件缺失；无法生成。");
        }
        trace.record(@"固定条件已加载");
        const TensorStats initialStats = tensorStats(hidden);
        if (!initialStats.finite) {
            NSString *message = [NSString stringWithFormat:@"初始 latent 含非有限值，位置 %zu。",
                                  initialStats.invalidIndex];
            trace.record(message);
            return failure(message);
        }
        trace.record([NSString stringWithFormat:@"初始 latent 峰值 %.8g", initialStats.peak]);

        NSString *ditPath = [directory stringByAppendingPathComponent:@"dit_w4.mnn"];
        if (![NSFileManager.defaultManager fileExistsAtPath:[directory stringByAppendingPathComponent:@"dit_w4.mnn.weight"]]) {
            trace.record(@"失败：DiT 权重缺失");
            return failure(@"DiT 外部权重缺失。");
        }
        {
            trace.record(@"开始创建 DiT 会话");
            auto dit = openGraph(ditPath, trace);
            if (!dit) return failure(@"无法创建 DiT MNN 会话；可能是内存不足或算子不支持。");
            trace.record(@"DiT 会话已创建");
            if (!setInput(*dit, "encoder_hidden_states", encoder.data(), encoder.size()) ||
                !setInput(*dit, "context_latents", context.data(), context.size())) {
                trace.record(@"失败：DiT 条件输入不匹配");
                return failure(@"DiT 固定条件输入形状不匹配。");
            }
            std::vector<float> velocity;
            for (size_t step = 0; step < kTimesteps.size(); ++step) {
                if (!trace.hasHeadroom()) {
                    trace.record(@"安全停止：剩余进程内存低于 512 MiB");
                    return failure(@"剩余进程内存低于 512 MiB，已停止以避免闪退。请分享诊断日志。");
                }
                trace.record([NSString stringWithFormat:@"DiT %zu/8 开始", step + 1]);
                float time = kTimesteps[step];
                if (!setInput(*dit, "hidden_states", hidden.data(), hidden.size()) ||
                    !setInput(*dit, "timestep", &time, 1) ||
                    !setInput(*dit, "timestep_r", &time, 1)) {
                    return failure(@"DiT 采样输入形状不匹配。");
                }
                const CFAbsoluteTime runStarted = CFAbsoluteTimeGetCurrent();
                MNN::ErrorCode code = dit->interpreter->runSession(dit->session);
                const CFAbsoluteTime runEnded = CFAbsoluteTimeGetCurrent();
                const bool readOK = code == MNN::NO_ERROR &&
                    readOutput(*dit, "denoised", latentCount, velocity);
                const CFAbsoluteTime readEnded = CFAbsoluteTimeGetCurrent();
                if (!readOK) {
                    trace.record([NSString stringWithFormat:@"DiT %zu/8 失败", step + 1]);
                    return failure([NSString stringWithFormat:@"DiT 第 %zu/8 步失败（MNN %d）。", step + 1, static_cast<int>(code)]);
                }
                const TensorStats velocityStats = tensorStats(velocity);
                if (!velocityStats.finite) {
                    NSString *message = [NSString stringWithFormat:@"DiT %zu/8 输出含非有限值，位置 %zu。",
                                          step + 1, velocityStats.invalidIndex];
                    trace.record(message);
                    return failure(message);
                }
                float next = step + 1 < kTimesteps.size() ? kTimesteps[step + 1] : 0.0f;
                float delta = time - next;
                for (size_t index = 0; index < latentCount; ++index) hidden[index] -= velocity[index] * delta;
                const TensorStats hiddenStats = tensorStats(hidden);
                if (!hiddenStats.finite) {
                    NSString *message = [NSString stringWithFormat:@"DiT %zu/8 更新后 latent 含非有限值，位置 %zu。",
                                          step + 1, hiddenStats.invalidIndex];
                    trace.record(message);
                    return failure(message);
                }
                trace.record([NSString stringWithFormat:
                    @"DiT %zu/8 完成；run=%.3f 秒 read=%.3f 秒 输出峰值 %.8g latent 峰值 %.8g",
                    step + 1, runEnded - runStarted, readEnded - runEnded,
                    velocityStats.peak, hiddenStats.peak]);
            }
        }
        trace.record(@"DiT 会话已释放；开始 VAE");

        std::vector<float> audio(static_cast<size_t>(2) * kSamples);
        int written = 0;
        {
            // One fixed 192-frame graph is reused for all chunks. Keeping the
            // 64-frame overlap avoids loading the larger 256-frame graph.
            trace.record(@"开始创建唯一的 VAE 192 帧会话");
            auto vae = openGraph([directory stringByAppendingPathComponent:@"vae_decoder_192_w4.mnn"], trace);
            if (!vae) return failure(@"无法加载 VAE 192 帧模型。");
            trace.record(@"VAE 192 帧会话已创建");
            std::vector<float> latent(static_cast<size_t>(kLatentChannels) * kVaeWindowFrames);
            std::vector<float> decoded;
            for (int chunk = 0; chunk < kVaeChunks; ++chunk) {
                if (!trace.hasHeadroom()) {
                    trace.record(@"安全停止：剩余进程内存低于 512 MiB");
                    return failure(@"剩余进程内存低于 512 MiB，已停止以避免闪退。请分享诊断日志。");
                }
                trace.record([NSString stringWithFormat:@"VAE %d/%d 开始", chunk + 1, kVaeChunks]);
                const int coreStart = chunk * kVaeCoreFrames;
                const int coreEnd = std::min(coreStart + kVaeCoreFrames, kLatentFrames);
                const int winStart = std::clamp(coreStart - kVaeOverlapFrames,
                                                0, kLatentFrames - kVaeWindowFrames);
                const int winEnd = winStart + kVaeWindowFrames;
                for (int channel = 0; channel < kLatentChannels; ++channel) {
                    for (int frame = winStart; frame < winEnd; ++frame) {
                        latent[channel * kVaeWindowFrames + frame - winStart] =
                            hidden[frame * kLatentChannels + channel];
                    }
                }
                const TensorStats latentStats = tensorStats(latent);
                if (!latentStats.finite) {
                    NSString *message = [NSString stringWithFormat:@"VAE %d/%d 输入含非有限值，位置 %zu。",
                                          chunk + 1, kVaeChunks, latentStats.invalidIndex];
                    trace.record(message);
                    return failure(message);
                }
                trace.record([NSString stringWithFormat:@"VAE %d/%d 输入峰值 %.8g",
                              chunk + 1, kVaeChunks, latentStats.peak]);
                if (!setInput(*vae, "latents", latent.data(), latent.size())) {
                    return failure(@"VAE 192 帧输入不匹配。");
                }
                MNN::ErrorCode code = vae->interpreter->runSession(vae->session);
                if (code != MNN::NO_ERROR ||
                    !readOutput(*vae, "waveform", static_cast<size_t>(2) * kVaeWindowFrames * kUpsample, decoded)) {
                    trace.record([NSString stringWithFormat:@"VAE %d/%d 失败", chunk + 1, kVaeChunks]);
                    return failure([NSString stringWithFormat:@"VAE 第 %d/%d 段失败（MNN %d）。",
                                    chunk + 1, kVaeChunks, static_cast<int>(code)]);
                }
                const int coreSamples = (coreEnd - coreStart) * kUpsample;
                const int offset = (coreStart - winStart) * kUpsample;
                float corePeak = 0.0f;
                for (int channel = 0; channel < 2; ++channel) {
                    const float *source = decoded.data() + channel * kVaeWindowFrames * kUpsample + offset;
                    for (int sample = 0; sample < coreSamples; ++sample) {
                        if (!std::isfinite(source[sample])) {
                            NSString *message = [NSString stringWithFormat:
                                @"VAE %d/%d 输出含非有限值，声道 %d，段内样本 %d。",
                                chunk + 1, kVaeChunks, channel, sample];
                            trace.record(message);
                            return failure(message);
                        }
                        corePeak = std::max(corePeak, std::abs(source[sample]));
                    }
                    float *target = audio.data() + channel * kSamples + written;
                    std::memcpy(target, source, static_cast<size_t>(coreSamples) * sizeof(float));
                }
                written += coreSamples;
                trace.record([NSString stringWithFormat:@"VAE %d/%d 完成；输出峰值 %.8g",
                              chunk + 1, kVaeChunks, corePeak]);
            }
        }
        if (written != kSamples) return failure(@"VAE 拼接后的音频长度不等于 30 秒。");
        NSString *saveError = nil;
        NSURL *target = saveWAV(audio, &saveError);
        if (!target) {
            NSString *message = saveError ?: @"写入 WAV 失败，系统未提供原因。";
            trace.record(message);
            return failure(message);
        }
        trace.record(@"WAV 已保存；生成完成");
        return @{ @"success": @YES, @"message": @"iPhone 已完成 DiT 八步及 VAE 十二段推理。歌词条件为预计算固定样例。",
                  @"wavPath": target.path, @"elapsedSeconds": @(CFAbsoluteTimeGetCurrent() - started),
                  @"audioSeconds": @30.0, @"backend": @"MNN Metal 请求 / CPU 回退可能" };
    }
}

@end
