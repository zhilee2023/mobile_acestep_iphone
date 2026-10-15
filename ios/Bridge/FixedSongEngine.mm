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
#include <utility>
#include <vector>
#include <unistd.h>

namespace {
constexpr int kLatentFrames = 750;
constexpr int kLatentChannels = 64;
constexpr int kSampleRate = 48000;
constexpr int kUpsample = 1920;
constexpr int kSamples = kLatentFrames * kUpsample;
constexpr int kVaeWindowFrames = 192;
constexpr size_t kMinimumAvailableBytes = 512ULL * 1024 * 1024;
// The iPhone's 192-frame VAE raised footprint by ~1.2 GB after DiT release.
// Leave a further ~0.5 GB margin before creating its session.
constexpr size_t kVaeRequiredAvailableBytes = 1700000000ULL;
constexpr std::array<float, 8> kTimesteps = {1.0f, 0.875f, 0.75f, 0.625f,
                                              0.5f, 0.375f, 0.25f, 0.125f};

NSDictionary<NSString *, id> *failure(NSString *message) {
    return @{ @"success": @NO, @"message": message };
}

struct InferenceTrace {
    FILE *file = nullptr;
    uint64_t sampledPeak = 0;

    InferenceTrace(bool twoStepProbe, bool highPrecision, bool optimizedVAE,
                   NSString *customFilename = nil, NSString *customOpening = nil) {
        NSURL *documents = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory
                                                                 inDomains:NSUserDomainMask].firstObject;
        NSURL *directory = [documents URLByAppendingPathComponent:@"Generated" isDirectory:YES];
        if (![NSFileManager.defaultManager createDirectoryAtURL:directory
                                    withIntermediateDirectories:YES attributes:nil error:nil]) return;
        NSURL *url = [directory URLByAppendingPathComponent:customFilename ?: @"last-inference.log"];
        file = std::fopen(url.fileSystemRepresentation, "w");
        record(customOpening ?: (twoStepProbe ? (highPrecision ? @"开始：DiT 两步诊断；Metal High、低内存模式；不运行 VAE"
                                             : @"开始：DiT 两步诊断；Metal Low、低内存模式；不运行 VAE")
                            : (optimizedVAE ? @"开始：30 秒固定样例；DiT Metal High、VAE Metal Low；优化串行／VAE 六段"
                                            : @"开始：30 秒固定样例；DiT Metal High、VAE Metal Low、低内存模式")));
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
        sampledPeak = std::max(sampledPeak, footprint);
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

std::unique_ptr<Graph> openGraph(NSString *path, InferenceTrace &trace,
                                 MNN::BackendConfig::PrecisionMode precision,
                                 MNNForwardType backendType = MNN_FORWARD_METAL,
                                 bool noTune = false) {
    auto graph = std::make_unique<Graph>();
    const CFAbsoluteTime loadStarted = CFAbsoluteTimeGetCurrent();
    graph->interpreter = MNN::Interpreter::createFromFile(path.fileSystemRepresentation);
    trace.record([NSString stringWithFormat:@"%@：读取模型 %.3f 秒", path.lastPathComponent,
                  CFAbsoluteTimeGetCurrent() - loadStarted]);
    if (!graph->interpreter) return nullptr;
    MNN::BackendConfig backend;
    backend.precision = precision;
    backend.memory = MNN::BackendConfig::Memory_Low;
    MNN::ScheduleConfig config;
    config.type = backendType;
    if (backendType == MNN_FORWARD_CPU) config.numThread = 4;
    if (backendType == MNN_FORWARD_METAL && noTune) config.mode = MNN_GPU_TUNING_NONE;
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

NSURL *saveWAV(const std::vector<float> &audio, NSString **reason, NSString *prefix) {
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
    NSString *filename = [NSString stringWithFormat:@"%@-%@.wav", prefix, [[NSUUID UUID] UUIDString]];
    NSURL *target = [directory URLByAppendingPathComponent:filename];
    NSData *data = [NSData dataWithBytesNoCopy:wave.data() length:wave.size() freeWhenDone:NO];
    NSError *writeError = nil;
    if (![data writeToURL:target options:NSDataWritingAtomic error:&writeError]) {
        if (reason) *reason = [NSString stringWithFormat:@"写入 WAV 失败：%@", writeError.localizedDescription];
        return nil;
    }
    return target;
}

struct VaeDiagnosticResult {
    bool ok = false;
    double loadSeconds = 0;
    double createSeconds = 0;
    double resizeSeconds = 0;
    double firstSeconds = 0;
    double secondSeconds = 0;
    uint64_t peakBytes = 0;
    int mainBackend = -1;
    std::vector<float> output;
};

bool waitForVaeHeadroom(InferenceTrace &trace, NSString *name) {
    constexpr size_t required = 2400000000ULL;
    int consecutivePasses = 0;
    const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    while (CFAbsoluteTimeGetCurrent() - started < 10.0) {
        const size_t available = trace.availableBytes();
        consecutivePasses = available >= required ? consecutivePasses + 1 : 0;
        if (consecutivePasses >= 2) {
            trace.record([NSString stringWithFormat:@"%@：内存门槛通过，等待 %.3f 秒", name,
                          CFAbsoluteTimeGetCurrent() - started]);
            return true;
        }
        usleep(250000);
    }
    trace.record([NSString stringWithFormat:@"%@：可用内存未恢复到 2400 MB，停止测试", name]);
    return false;
}

bool loadVaeDiagnosticLatent(std::vector<float> &latent) {
    std::vector<float> finalLatent;
    if (!loadFixture(@"vae-final-latent", static_cast<size_t>(kLatentFrames) * kLatentChannels, finalLatent)) {
        return false;
    }
    constexpr int windowStart = 256;
    latent.resize(static_cast<size_t>(kLatentChannels) * kVaeWindowFrames);
    for (int channel = 0; channel < kLatentChannels; ++channel) {
        for (int frame = 0; frame < kVaeWindowFrames; ++frame) {
            latent[channel * kVaeWindowFrames + frame] =
                finalLatent[(windowStart + frame) * kLatentChannels + channel];
        }
    }
    return tensorStats(latent).finite;
}

VaeDiagnosticResult runVaeDiagnosticCase(NSString *path, NSString *name,
                                         MNNForwardType backendType,
                                         MNN::BackendConfig::PrecisionMode precision,
                                         bool deferred, int gpuMode,
                                         const std::vector<float> &latent,
                                         InferenceTrace &trace) {
    VaeDiagnosticResult result;
    if (!waitForVaeHeadroom(trace, name)) return result;
    trace.sampledPeak = 0;
    trace.record([NSString stringWithFormat:@"%@ 开始：%@，%@，Memory Low；相同 192 帧输入",
                  name, backendType == MNN_FORWARD_CPU ? @"CPU" : @"Metal",
                  deferred ? @"延迟 resize" : @"常规 createSession"]);
    {
        Graph graph;
        const CFAbsoluteTime loadStarted = CFAbsoluteTimeGetCurrent();
        graph.interpreter = MNN::Interpreter::createFromFile(path.fileSystemRepresentation);
        result.loadSeconds = CFAbsoluteTimeGetCurrent() - loadStarted;
        trace.record([NSString stringWithFormat:@"%@：createFromFile %.3f 秒", name, result.loadSeconds]);
        if (!graph.interpreter) return result;
        if (deferred) graph.interpreter->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        MNN::BackendConfig backend;
        backend.precision = precision;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = backendType;
        config.mode = backendType == MNN_FORWARD_METAL ? gpuMode : 4;
        config.backupType = MNN_FORWARD_CPU;
        config.backendConfig = &backend;
        const CFAbsoluteTime createStarted = CFAbsoluteTimeGetCurrent();
        graph.session = graph.interpreter->createSession(config);
        result.createSeconds = CFAbsoluteTimeGetCurrent() - createStarted;
        trace.record([NSString stringWithFormat:@"%@：createSession %.3f 秒%@", name,
                      result.createSeconds, deferred ? @"（未包含显式 resize）" : @"（包含初始 resize）"]);
        if (!graph.session) return result;
        if (deferred) {
            const CFAbsoluteTime resizeStarted = CFAbsoluteTimeGetCurrent();
            graph.interpreter->resizeSession(graph.session);
            result.resizeSeconds = CFAbsoluteTimeGetCurrent() - resizeStarted;
            trace.record([NSString stringWithFormat:@"%@：显式 resizeSession %.3f 秒", name, result.resizeSeconds]);
            int resizeStatus = -1;
            if (!graph.interpreter->getSessionInfo(graph.session, MNN::Interpreter::RESIZE_STATUS, &resizeStatus)
                || resizeStatus != 0) {
                trace.record([NSString stringWithFormat:@"%@：resize 状态 %d，停止", name, resizeStatus]);
                return result;
            }
        }
        int actualBackends[2] = {-1, -1};
        if (graph.interpreter->getSessionInfo(graph.session, MNN::Interpreter::BACKENDS, actualBackends)) {
            result.mainBackend = actualBackends[0];
            trace.record([NSString stringWithFormat:@"%@：主 pipeline backend=%d（CPU=0，Metal=1；不代表逐算子均在该后端）",
                          name, actualBackends[0]]);
            if (backendType == MNN_FORWARD_METAL && actualBackends[0] != MNN_FORWARD_METAL) {
                trace.record([NSString stringWithFormat:@"%@：未获得 Metal 主 pipeline，停止该组", name]);
                return result;
            }
        }
        graph.interpreter->releaseModel();
        for (int run = 0; run < 2; ++run) {
            const size_t needed = run == 0 ? 1200000000ULL : kMinimumAvailableBytes;
            if (trace.availableBytes() < needed) {
                trace.record([NSString stringWithFormat:@"%@：第 %d 次前向前内存不足 %.0f MB，安全停止",
                              name, run + 1, needed / 1000000.0]);
                return result;
            }
            if (!setInput(graph, "latents", latent.data(), latent.size())) {
                trace.record([NSString stringWithFormat:@"%@：输入形状不匹配", name]);
                return result;
            }
            const CFAbsoluteTime runStarted = CFAbsoluteTimeGetCurrent();
            const MNN::ErrorCode code = graph.interpreter->runSession(graph.session);
            const CFAbsoluteTime runEnded = CFAbsoluteTimeGetCurrent();
            std::vector<float> output;
            const bool readOK = code == MNN::NO_ERROR &&
                readOutput(graph, "waveform", static_cast<size_t>(2) * kVaeWindowFrames * kUpsample, output);
            const CFAbsoluteTime readEnded = CFAbsoluteTimeGetCurrent();
            if (!readOK) {
                trace.record([NSString stringWithFormat:@"%@：第 %d 次前向失败，MNN=%d", name, run + 1, static_cast<int>(code)]);
                return result;
            }
            const TensorStats stats = tensorStats(output);
            if (!stats.finite) {
                trace.record([NSString stringWithFormat:@"%@：第 %d 次输出非有限值，位置 %zu", name, run + 1, stats.invalidIndex]);
                return result;
            }
            if (stats.peak < 1e-6f) {
                trace.record([NSString stringWithFormat:@"%@：第 %d 次输出接近静音，停止", name, run + 1]);
                return result;
            }
            const double elapsed = readEnded - runStarted;
            if (run == 0) result.firstSeconds = elapsed;
            else result.secondSeconds = elapsed;
            result.output = std::move(output);
            trace.record([NSString stringWithFormat:@"%@：第 %d 次 run=%.3f 秒，read/sync=%.3f 秒，总计 %.3f 秒，峰值 %.8g",
                          name, run + 1, runEnded - runStarted, readEnded - runEnded, elapsed, stats.peak]);
        }
        result.ok = true;
        result.peakBytes = trace.sampledPeak;
    }
    trace.record([NSString stringWithFormat:@"%@：会话已释放；采样峰值 %.0f MB", name,
                  result.peakBytes / 1000000.0]);
    return result;
}

double relativeL2(const std::vector<float> &actual, const std::vector<float> &reference) {
    if (actual.size() != reference.size() || actual.empty()) return NAN;
    double error = 0.0;
    double energy = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double difference = static_cast<double>(actual[i]) - reference[i];
        error += difference * difference;
        energy += static_cast<double>(reference[i]) * reference[i];
    }
    return std::sqrt(error / std::max(energy, 1e-20));
}
} // namespace

@interface FixedSongEngine ()
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory twoStepProbe:(BOOL)twoStepProbe
                                   highPrecision:(BOOL)highPrecision optimizedVAE:(BOOL)optimizedVAE
                                        condition:(NSData *)condition cpuVAE:(BOOL)cpuVAE noTuneVAE:(BOOL)noTuneVAE;
@end

@implementation FixedSongEngine

+ (NSDictionary<NSString *, id> *)diagnoseVAEAtDirectory:(NSString *)directory {
    @autoreleasepool {
        InferenceTrace trace(false, false, false, @"vae-diagnostic.log",
                             @"开始：VAE 192 帧五组独立会话；CPU High/Low、Metal Low Wide/NoTune；记录 create/resize/前向/同步/内存；不生成歌曲");
        const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        NSString *path = [directory stringByAppendingPathComponent:@"vae_decoder_192_w4.mnn"];
        if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
            trace.record(@"失败：VAE 模型文件缺失");
            return failure(@"VAE 192 帧模型文件缺失。");
        }
        std::vector<float> latent;
        if (!loadVaeDiagnosticLatent(latent)) {
            trace.record(@"失败：诊断 latent 缺失");
            return failure(@"VAE 诊断输入缺失，请重新安装诊断版。");
        }
        constexpr int windowStart = 256;
        const TensorStats inputStats = tensorStats(latent);
        if (!inputStats.finite) return failure(@"VAE 诊断输入含非有限值。");
        trace.record([NSString stringWithFormat:@"固定输入：DiT 后 latent 第 %d–%d 帧，峰值 %.8g",
                      windowStart, windowStart + kVaeWindowFrames - 1, inputStats.peak]);

        const VaeDiagnosticResult cpuHigh = runVaeDiagnosticCase(path, @"CPU High / 延迟 resize",
            MNN_FORWARD_CPU, MNN::BackendConfig::Precision_High, true, 0, latent, trace);
        if (!cpuHigh.ok) return failure(@"CPU High 基线未完成；请分享 VAE 诊断日志。");
        const VaeDiagnosticResult cpuLow = runVaeDiagnosticCase(path, @"CPU Low / 常规会话",
            MNN_FORWARD_CPU, MNN::BackendConfig::Precision_Low, false, 0, latent, trace);
        if (!cpuLow.ok) return failure(@"CPU Low 测试未完成；请分享 VAE 诊断日志。");
        trace.record([NSString stringWithFormat:@"CPU Low 相对 CPU High 输出 L2 = %.8g",
                      relativeL2(cpuLow.output, cpuHigh.output)]);

        const VaeDiagnosticResult metalDeferred = runVaeDiagnosticCase(path, @"Metal Low / Wide / 延迟 resize",
            MNN_FORWARD_METAL, MNN::BackendConfig::Precision_Low, true, MNN_GPU_TUNING_WIDE, latent, trace);
        if (!metalDeferred.ok) return failure(@"Metal Low 延迟 resize 测试未完成；请分享 VAE 诊断日志。");
        trace.record([NSString stringWithFormat:@"Metal Low 延迟 resize 相对 CPU Low 输出 L2 = %.8g",
                      relativeL2(metalDeferred.output, cpuLow.output)]);

        const VaeDiagnosticResult metalDirect = runVaeDiagnosticCase(path, @"Metal Low / Wide / 常规会话重复",
            MNN_FORWARD_METAL, MNN::BackendConfig::Precision_Low, false, MNN_GPU_TUNING_WIDE, latent, trace);
        if (!metalDirect.ok) return failure(@"Metal Low 常规会话复测未完成；请分享 VAE 诊断日志。");
        const double sameBackendL2 = relativeL2(metalDirect.output, metalDeferred.output);
        trace.record([NSString stringWithFormat:@"Metal Low 常规 vs 延迟 resize 输出 L2 = %.8g", sameBackendL2]);
        const VaeDiagnosticResult metalNoTune = runVaeDiagnosticCase(path, @"Metal Low / NoTune / 常规会话",
            MNN_FORWARD_METAL, MNN::BackendConfig::Precision_Low, false, MNN_GPU_TUNING_NONE, latent, trace);
        if (!metalNoTune.ok) return failure(@"Metal Low 禁用调优测试未完成；请分享 VAE 诊断日志。");
        const double noTuneL2 = relativeL2(metalNoTune.output, metalDirect.output);
        trace.record([NSString stringWithFormat:@"Metal NoTune 相对 Wide 输出 L2 = %.8g；创建 %.3f vs %.3f 秒；第二次解码 %.3f vs %.3f 秒",
                      noTuneL2, metalNoTune.createSeconds, metalDirect.createSeconds,
                      metalNoTune.secondSeconds, metalDirect.secondSeconds]);
        trace.record([NSString stringWithFormat:
            @"诊断完成；总耗时 %.3f 秒；CPU High create+resize=%.3f；CPU Low create=%.3f；Metal Low Wide 延迟 create=%.3f resize=%.3f；Wide 常规 create=%.3f；NoTune create=%.3f；NoTune 对 Wide L2=%.8g",
            CFAbsoluteTimeGetCurrent() - started, cpuHigh.createSeconds + cpuHigh.resizeSeconds,
            cpuLow.createSeconds, metalDeferred.createSeconds, metalDeferred.resizeSeconds,
            metalDirect.createSeconds, metalNoTune.createSeconds, noTuneL2]);
        return @{ @"success": @YES,
                  @"message": @"VAE 五组 CPU/Metal 对照完成；请分享 VAE 诊断日志。此测试不生成音频，也不改变歌词。",
                  @"elapsedSeconds": @(CFAbsoluteTimeGetCurrent() - started) };
    }
}

+ (NSDictionary<NSString *, id> *)diagnoseVAENoTuneAtDirectory:(NSString *)directory {
    @autoreleasepool {
        InferenceTrace trace(false, false, false, @"vae-notune-diagnostic.log",
                             @"开始：独立 VAE 192 帧 Metal Low / NoTune；固定输入、两次前向；不运行其他四组，不生成歌曲");
        NSString *path = [directory stringByAppendingPathComponent:@"vae_decoder_192_w4.mnn"];
        if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
            trace.record(@"失败：VAE 模型文件缺失");
            return failure(@"VAE 192 帧模型文件缺失。");
        }
        std::vector<float> latent;
        if (!loadVaeDiagnosticLatent(latent)) {
            trace.record(@"失败：诊断 latent 缺失或非有限值");
            return failure(@"VAE 诊断输入缺失或异常，请重新安装诊断版。");
        }
        trace.record([NSString stringWithFormat:@"固定输入：DiT 后 latent 第 256–447 帧，峰值 %.8g",
                      tensorStats(latent).peak]);
        const VaeDiagnosticResult result = runVaeDiagnosticCase(
            path, @"Metal Low / NoTune / 独立会话", MNN_FORWARD_METAL,
            MNN::BackendConfig::Precision_Low, false, MNN_GPU_TUNING_NONE, latent, trace);
        if (!result.ok) {
            trace.record(@"NoTune 未完成；未做性能或音质结论");
            return failure(@"NoTune 测试未完成；请分享独立诊断日志。若内存不足，请关闭并重新打开 App 后再测。");
        }
        trace.record([NSString stringWithFormat:@"NoTune 完成：创建 %.3f 秒；首次解码 %.3f 秒；第二次解码 %.3f 秒；主 backend=%d；输出含有限值，峰值 %.8g；采样峰值 %.0f MB",
                      result.createSeconds, result.firstSeconds, result.secondSeconds,
                      result.mainBackend, tensorStats(result.output).peak, result.peakBytes / 1000000.0]);
        return @{ @"success": @YES,
                  @"message": @"独立 NoTune 诊断完成；请分享日志。输出有限且非静音，但尚未验证整首音质。" };
    }
}

+ (NSDictionary<NSString *, id> *)generateAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory twoStepProbe:NO highPrecision:YES optimizedVAE:NO condition:nil cpuVAE:NO noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)generateOptimizedAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory twoStepProbe:NO highPrecision:YES optimizedVAE:YES condition:nil cpuVAE:NO noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)generateOptimizedCpuVAEAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory twoStepProbe:NO highPrecision:YES optimizedVAE:YES condition:nil cpuVAE:YES noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)generateOptimizedNoTuneVAEAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory twoStepProbe:NO highPrecision:YES optimizedVAE:YES condition:nil cpuVAE:NO noTuneVAE:YES];
}

+ (NSDictionary<NSString *, id> *)generateWithCondition:(NSData *)condition atDirectory:(NSString *)directory {
    if (condition.length != static_cast<size_t>(181) * 2048 * sizeof(float)) {
        return failure(@"自定义歌词条件形状不正确。");
    }
    return [self runAtDirectory:directory twoStepProbe:NO highPrecision:YES optimizedVAE:YES condition:condition cpuVAE:NO noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)generateWithConditionCpuVAE:(NSData *)condition atDirectory:(NSString *)directory {
    if (condition.length != static_cast<size_t>(181) * 2048 * sizeof(float)) {
        return failure(@"歌词／风格条件形状不正确。");
    }
    return [self runAtDirectory:directory twoStepProbe:NO highPrecision:YES optimizedVAE:YES condition:condition cpuVAE:YES noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)probeTwoStepHighAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory twoStepProbe:YES highPrecision:YES optimizedVAE:NO condition:nil cpuVAE:NO noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)probeTwoStepLowAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory twoStepProbe:YES highPrecision:NO optimizedVAE:NO condition:nil cpuVAE:NO noTuneVAE:NO];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory twoStepProbe:(BOOL)twoStepProbe
                                   highPrecision:(BOOL)highPrecision optimizedVAE:(BOOL)optimizedVAE
                                        condition:(NSData *)condition cpuVAE:(BOOL)cpuVAE noTuneVAE:(BOOL)noTuneVAE {
    @autoreleasepool {
        InferenceTrace trace(twoStepProbe, highPrecision, optimizedVAE,
                             cpuVAE ? (condition ? @"last-inference-custom-cpu-vae.log" : @"last-inference-cpu-vae.log")
                                    : (noTuneVAE ? @"last-inference-notune-vae.log" : nil),
                             cpuVAE ? (condition ? @"开始：30 秒自定义歌词与风格；DiT Metal High → VAE CPU Low；VAE 六段"
                                                 : @"开始：30 秒固定样例；DiT Metal High → VAE CPU Low；VAE 六段")
                                    : (noTuneVAE ? @"开始：30 秒固定样例；DiT Metal High → VAE Metal Low NoTune；VAE 六段"
                                                 : (condition ? @"开始：30 秒自定义歌词；固定风格；DiT Metal High → VAE Metal Low；VAE 六段"
                                                 : nil)));
        const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        const size_t latentCount = static_cast<size_t>(kLatentFrames) * kLatentChannels;
        std::vector<float> hidden, encoder, context;
        if (!loadFixture(@"dit-hidden", latentCount, hidden) ||
            (!condition && !loadFixture(@"dit-encoder", 181 * 2048, encoder)) ||
            !loadFixture(@"dit-context", static_cast<size_t>(kLatentFrames) * 128, context)) {
            trace.record(@"失败：固定条件缺失");
            return failure(@"固定歌曲条件文件缺失；无法生成。");
        }
        if (condition) {
            encoder.resize(static_cast<size_t>(181) * 2048);
            std::memcpy(encoder.data(), condition.bytes, condition.length);
            if (!tensorStats(encoder).finite) return failure(@"自定义歌词条件包含非有限值。");
            trace.record(cpuVAE ? @"自定义歌词和风格条件已接入 DiT；无参考音频，timbre 使用默认固定条件"
                                : @"自定义歌词条件已接入 DiT；风格与 timbre 仍为固定样例");
        } else {
            trace.record(@"固定条件已加载");
        }
        const TensorStats initialStats = tensorStats(hidden);
        if (!initialStats.finite) {
            NSString *message = [NSString stringWithFormat:@"初始 latent 含非有限值，位置 %zu。",
                                  initialStats.invalidIndex];
            trace.record(message);
            return failure(message);
        }
        trace.record([NSString stringWithFormat:@"初始 latent 峰值 %.8g", initialStats.peak]);

        std::vector<float> cpuStep1, cpuStep2;
        if (twoStepProbe &&
            (!loadFixture(@"dit-w4-cpu-step1", latentCount, cpuStep1) ||
             !loadFixture(@"dit-w4-cpu-step2", latentCount, cpuStep2))) {
            trace.record(@"失败：本地 CPU/W4 两步参考数据缺失");
            return failure(@"本地 CPU/W4 两步参考数据缺失。请重新安装诊断版。");
        }

        NSString *ditPath = [directory stringByAppendingPathComponent:@"dit_w4.mnn"];
        if (![NSFileManager.defaultManager fileExistsAtPath:[directory stringByAppendingPathComponent:@"dit_w4.mnn.weight"]]) {
            trace.record(@"失败：DiT 权重缺失");
            return failure(@"DiT 外部权重缺失。");
        }
        {
            trace.record(@"开始创建 DiT 会话");
            auto dit = openGraph(ditPath, trace, highPrecision ? MNN::BackendConfig::Precision_High
                                                               : MNN::BackendConfig::Precision_Low);
            if (!dit) return failure(@"无法创建 DiT MNN 会话；可能是内存不足或算子不支持。");
            trace.record(@"DiT 会话已创建");
            trace.record(@"DiT 将在每步重新传入全部五个输入");
            std::vector<float> velocity;
            const size_t stepCount = twoStepProbe ? 2 : kTimesteps.size();
            for (size_t step = 0; step < stepCount; ++step) {
                if (!trace.hasHeadroom()) {
                    trace.record(@"安全停止：剩余进程内存低于 512 MiB");
                    return failure(@"剩余进程内存低于 512 MiB，已停止以避免闪退。请分享诊断日志。");
                }
                trace.record([NSString stringWithFormat:@"DiT %zu/8 开始", step + 1]);
                float time = kTimesteps[step];
                if (!setInput(*dit, "encoder_hidden_states", encoder.data(), encoder.size()) ||
                    !setInput(*dit, "context_latents", context.data(), context.size()) ||
                    !setInput(*dit, "hidden_states", hidden.data(), hidden.size()) ||
                    !setInput(*dit, "timestep", &time, 1) ||
                    !setInput(*dit, "timestep_r", &time, 1)) {
                    trace.record(@"失败：DiT 输入不匹配");
                    return failure(@"DiT 输入形状不匹配或传入失败。");
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
                if (twoStepProbe) {
                    const std::vector<float> &reference = step == 0 ? cpuStep1 : cpuStep2;
                    double squaredError = 0.0;
                    double squaredReference = 0.0;
                    for (size_t index = 0; index < latentCount; ++index) {
                        const double difference = static_cast<double>(velocity[index]) - reference[index];
                        squaredError += difference * difference;
                        squaredReference += static_cast<double>(reference[index]) * reference[index];
                    }
                    const double relativeL2 = std::sqrt(squaredError / std::max(squaredReference, 1e-20));
                    trace.record([NSString stringWithFormat:@"DiT %zu/2 相对本地 CPU/W4 输出 L2 = %.6g",
                                  step + 1, relativeL2]);
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
        if (twoStepProbe) {
            NSString *precisionName = highPrecision ? @"High" : @"Low";
            trace.record([NSString stringWithFormat:@"DiT Metal %@ 两步完成；诊断结束，未运行 VAE", precisionName]);
            return @{ @"success": @YES, @"message": [NSString stringWithFormat:@"DiT Metal %@ 两步输出均为有限值；请分享诊断日志。未运行 VAE，也未生成音频。", precisionName],
                      @"elapsedSeconds": @(CFAbsoluteTimeGetCurrent() - started) };
        }
        trace.record(@"DiT 会话已释放；开始 VAE");
        if (noTuneVAE) {
            const CFAbsoluteTime waitStarted = CFAbsoluteTimeGetCurrent();
            int consecutive = 0;
            while (CFAbsoluteTimeGetCurrent() - waitStarted < 10.0) {
                consecutive = trace.availableBytes() >= 2400000000ULL ? consecutive + 1 : 0;
                if (consecutive >= 2) break;
                usleep(250000);
            }
            trace.record([NSString stringWithFormat:@"NoTune VAE 内存等待 %.3f 秒；达标 %d/2",
                          CFAbsoluteTimeGetCurrent() - waitStarted, consecutive]);
            if (consecutive < 2) return failure(@"DiT 已完成，但 NoTune VAE 所需的 2400 MB 内存余量未恢复；已安全停止。");
        }
        if (trace.availableBytes() < kVaeRequiredAvailableBytes) {
            trace.record(@"安全停止：DiT 释放后可用内存不足 1700 MB，未创建 VAE 会话");
            return failure(@"DiT 已完成，但剩余内存不足以安全创建 VAE。请分享诊断日志。");
        }

        std::vector<float> audio(static_cast<size_t>(2) * kSamples);
        int written = 0;
        const int coreFrames = optimizedVAE ? 128 : 64;
        const int overlapFrames = (kVaeWindowFrames - coreFrames) / 2;
        const int vaeChunks = (kLatentFrames + coreFrames - 1) / coreFrames;
        {
            // Reuse one 192-frame graph. The optimized 128-frame core keeps
            // 32 frames of context on each side, validated against 64/64.
            trace.record(cpuVAE ? @"开始创建唯一的 VAE 192 帧 CPU Low 会话"
                                : (noTuneVAE ? @"开始创建唯一的 VAE 192 帧 Metal Low NoTune 会话"
                                             : @"开始创建唯一的 VAE 192 帧 Metal Low 会话"));
            auto vae = openGraph([directory stringByAppendingPathComponent:@"vae_decoder_192_w4.mnn"], trace,
                                 MNN::BackendConfig::Precision_Low,
                                 cpuVAE ? MNN_FORWARD_CPU : MNN_FORWARD_METAL, noTuneVAE);
            if (!vae) return failure(@"无法加载 VAE 192 帧模型。");
            int actualBackends[2] = {-1, -1};
            if (vae->interpreter->getSessionInfo(vae->session, MNN::Interpreter::BACKENDS, actualBackends)) {
                trace.record([NSString stringWithFormat:@"VAE 主 pipeline backend=%d（CPU Low 可为 13/CPU Extension；Metal=1）",
                              actualBackends[0]]);
                if (cpuVAE && actualBackends[0] != MNN_FORWARD_CPU && actualBackends[0] != 13) {
                    return failure(@"VAE CPU Low 对照未使用 CPU 后端，已停止。");
                }
                if (noTuneVAE && actualBackends[0] != MNN_FORWARD_METAL) {
                    return failure(@"VAE NoTune 未使用 Metal 主 pipeline，已停止。");
                }
            }
            trace.record(@"VAE 192 帧会话已创建");
            std::vector<float> latent(static_cast<size_t>(kLatentChannels) * kVaeWindowFrames);
            std::vector<float> decoded;
            for (int chunk = 0; chunk < vaeChunks; ++chunk) {
                if (!trace.hasHeadroom()) {
                    trace.record(@"安全停止：剩余进程内存低于 512 MiB");
                    return failure(@"剩余进程内存低于 512 MiB，已停止以避免闪退。请分享诊断日志。");
                }
                trace.record([NSString stringWithFormat:@"VAE %d/%d 开始", chunk + 1, vaeChunks]);
                const int coreStart = chunk * coreFrames;
                const int coreEnd = std::min(coreStart + coreFrames, kLatentFrames);
                const int winStart = std::clamp(coreStart - overlapFrames,
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
                                          chunk + 1, vaeChunks, latentStats.invalidIndex];
                    trace.record(message);
                    return failure(message);
                }
                trace.record([NSString stringWithFormat:@"VAE %d/%d 输入峰值 %.8g",
                              chunk + 1, vaeChunks, latentStats.peak]);
                if (!setInput(*vae, "latents", latent.data(), latent.size())) {
                    return failure(@"VAE 192 帧输入不匹配。");
                }
                MNN::ErrorCode code = vae->interpreter->runSession(vae->session);
                if (code != MNN::NO_ERROR ||
                    !readOutput(*vae, "waveform", static_cast<size_t>(2) * kVaeWindowFrames * kUpsample, decoded)) {
                    trace.record([NSString stringWithFormat:@"VAE %d/%d 失败", chunk + 1, vaeChunks]);
                    return failure([NSString stringWithFormat:@"VAE 第 %d/%d 段失败（MNN %d）。",
                                    chunk + 1, vaeChunks, static_cast<int>(code)]);
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
                                chunk + 1, vaeChunks, channel, sample];
                            trace.record(message);
                            return failure(message);
                        }
                        corePeak = std::max(corePeak, std::abs(source[sample]));
                    }
                    float *target = audio.data() + channel * kSamples + written;
                    std::memcpy(target, source, static_cast<size_t>(coreSamples) * sizeof(float));
                }
                written += coreSamples;
                trace.record([NSString stringWithFormat:@"VAE %d/%d 完成；输出峰值 %.8g；已解码 %.2f 秒音频；从生成开始 %.3f 秒",
                              chunk + 1, vaeChunks, corePeak, static_cast<double>(coreEnd) / 25.0,
                              CFAbsoluteTimeGetCurrent() - started]);
            }
        }
        if (written != kSamples) return failure(@"VAE 拼接后的音频长度不等于 30 秒。");
        NSString *saveError = nil;
        NSString *filenamePrefix = condition ? (cpuVAE ? @"custom-song-30s-w4-cpu-vae" : @"custom-lyrics-30s-w4-metal-vae")
                                             : (cpuVAE ? @"fixed-30s-w4-cpu-vae" : (noTuneVAE ? @"fixed-30s-w4-metal-notune-vae" : @"fixed-30s-w4-metal-vae"));
        NSURL *target = saveWAV(audio, &saveError, filenamePrefix);
        if (!target) {
            NSString *message = saveError ?: @"写入 WAV 失败，系统未提供原因。";
            trace.record(message);
            return failure(message);
        }
        trace.record(@"WAV 已保存；生成完成");
        return @{ @"success": @YES, @"message": [NSString stringWithFormat:@"iPhone 已完成 DiT 八步及 VAE %d 段推理。%@；歌词条件为%@。", vaeChunks,
                       cpuVAE ? @"VAE CPU Low" : (noTuneVAE ? @"VAE Metal Low NoTune" : @"VAE Metal Low"),
                       condition ? (cpuVAE ? @"本机编码的新歌词和新风格" : @"本机编码的新歌词") : @"预计算固定样例"],
                  @"wavPath": target.path, @"elapsedSeconds": @(CFAbsoluteTimeGetCurrent() - started),
                  @"audioSeconds": @30.0,
                  @"backend": cpuVAE ? @"DiT Metal 请求 / VAE CPU Low"
                                     : (noTuneVAE ? @"DiT Metal 请求 / VAE Metal NoTune" : @"MNN Metal 请求 / CPU 回退可能") };
    }
}

@end
