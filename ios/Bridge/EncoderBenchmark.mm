#import "EncoderBenchmark.h"
#import "FixedSongEngine.h"
#import <MNN/Interpreter.hpp>
#import <mach/mach.h>
#import <os/proc.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unistd.h>

namespace {
constexpr size_t kTokens = 107;
constexpr size_t kHiddenSize = 1024;
constexpr size_t kWarmRuns = 3;
constexpr size_t kMinimumAvailableBytes = 512ULL * 1024 * 1024;
// The existing iPhone DiT High probe reached ~1.97 GB footprint from a ~72 MB
// baseline. Require at least ~1.9 GB for its growth plus ~0.5 GB margin.
constexpr size_t kSerialDiTRequiredBytes = 2400000000ULL;

NSDictionary<NSString *, id> *failure(NSString *message) {
    return @{ @"success": @NO, @"message": message };
}

struct Trace {
    FILE *file = nullptr;
    uint64_t sampledPeak = 0;

    Trace(const char *filename = "encoder-benchmark.log",
          NSString *opening = @"开始：Qwen3-Embedding W4；107-token 固定输入；Metal Low／低内存；无 KV-cache decode") {
        NSURL *documents = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory
                                                                 inDomains:NSUserDomainMask].firstObject;
        NSURL *directory = [documents URLByAppendingPathComponent:@"Generated" isDirectory:YES];
        if (![NSFileManager.defaultManager createDirectoryAtURL:directory
                                    withIntermediateDirectories:YES attributes:nil error:nil]) return;
        NSURL *url = [directory URLByAppendingPathComponent:[NSString stringWithUTF8String:filename]];
        file = std::fopen(url.fileSystemRepresentation, "w");
        record(opening);
    }

    ~Trace() { if (file) std::fclose(file); }

    size_t availableBytes() const { return os_proc_available_memory(); }

    void record(NSString *phase) {
        task_vm_info_data_t info = {};
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        uint64_t footprint = 0;
        if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count)
                == KERN_SUCCESS) footprint = info.phys_footprint;
        sampledPeak = std::max(sampledPeak, footprint);
        if (!file) return;
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

bool setInput(Graph &graph, const NSData *idsData) {
    MNN::Tensor *input = graph.interpreter->getSessionInput(graph.session, "input_ids");
    if (!input || idsData.length != kTokens * sizeof(int64_t)) return false;
    MNN::Tensor host(input, input->getDimensionType(), true);
    const int64_t *ids = static_cast<const int64_t *>(idsData.bytes);
    if (input->getType().bits == 64 && host.usize() == kTokens * sizeof(int64_t)) {
        if (!host.host<int64_t>()) return false;
        std::memcpy(host.host<int64_t>(), ids, idsData.length);
    } else if (input->getType().bits == 32 && host.usize() == kTokens * sizeof(int32_t)) {
        if (!host.host<int32_t>()) return false;
        for (size_t index = 0; index < kTokens; ++index) {
            if (ids[index] < INT32_MIN || ids[index] > INT32_MAX) return false;
            host.host<int32_t>()[index] = static_cast<int32_t>(ids[index]);
        }
    } else {
        return false;
    }
    if (input->copyFromHostTensor(&host)) return true;
    if (input->getType().bits == 64 && input->host<int64_t>()) {
        std::memcpy(input->host<int64_t>(), host.host<int64_t>(), host.usize());
        return true;
    }
    if (input->getType().bits == 32 && input->host<int32_t>()) {
        std::memcpy(input->host<int32_t>(), host.host<int32_t>(), host.usize());
        return true;
    }
    return false;
}

bool readOutput(Graph &graph, float &peak) {
    MNN::Tensor *output = graph.interpreter->getSessionOutput(graph.session, "last_hidden_state");
    if (!output) return false;
    MNN::Tensor host(output, output->getDimensionType(), true);
    if (!host.host<float>() || host.usize() != kTokens * kHiddenSize * sizeof(float)) return false;
    if (!output->copyToHostTensor(&host)) {
        if (!output->host<float>()) return false;
        std::memcpy(host.host<float>(), output->host<float>(), host.usize());
    }
    peak = 0.0f;
    for (size_t index = 0; index < kTokens * kHiddenSize; ++index) {
        const float value = host.host<float>()[index];
        if (!std::isfinite(value)) return false;
        peak = std::max(peak, std::abs(value));
    }
    return true;
}
} // namespace

@interface EncoderBenchmark ()
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory warmRuns:(size_t)warmRuns;
+ (NSDictionary<NSString *, id> *)runFullSerialSongAtDirectory:(NSString *)directory optimized:(BOOL)optimized
                                                cpuVAE:(BOOL)cpuVAE;
@end

@implementation EncoderBenchmark

+ (NSDictionary<NSString *, id> *)runFullSerialSongAtDirectory:(NSString *)directory {
    return [self runFullSerialSongAtDirectory:directory optimized:NO cpuVAE:NO];
}

+ (NSDictionary<NSString *, id> *)runOptimizedFullSerialAtDirectory:(NSString *)directory {
    return [self runFullSerialSongAtDirectory:directory optimized:YES cpuVAE:NO];
}

+ (NSDictionary<NSString *, id> *)runOptimizedCpuVaeSerialAtDirectory:(NSString *)directory {
    return [self runFullSerialSongAtDirectory:directory optimized:YES cpuVAE:YES];
}

+ (NSDictionary<NSString *, id> *)runFullSerialSongAtDirectory:(NSString *)directory optimized:(BOOL)optimized
                                                cpuVAE:(BOOL)cpuVAE {
    Trace trace(cpuVAE ? "encoder-cpu-vae-serial.log"
                       : (optimized ? "encoder-optimized-serial.log" : "encoder-full-serial.log"),
                cpuVAE ? @"开始：Encoder 一次前向 → 释放 → 自适应内存门槛 → DiT Metal 八步 → VAE CPU Low 六段 → 30 秒 WAV；固定条件"
                       : (optimized ? @"开始：Encoder 一次前向 → 释放 → 自适应内存门槛 → DiT 八步 → VAE 六段 → 30 秒 WAV；固定输入"
                                    : @"开始：Encoder 一次前向 → 释放 → DiT 八步 → VAE 十二段 → 30 秒 WAV；固定输入"));
    const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    NSDictionary<NSString *, id> *encoder = [self runAtDirectory:directory warmRuns:0];
    trace.record(@"Encoder 测试函数已返回；立即采样内存");
    if (![encoder[@"success"] boolValue]) {
        trace.record([NSString stringWithFormat:@"停止：Encoder 失败；%@", encoder[@"message"] ?: @"未知原因"]);
        return encoder;
    }
    CFAbsoluteTime recoveryStarted = CFAbsoluteTimeGetCurrent();
    if (optimized) {
        int consecutivePasses = 0;
        while (CFAbsoluteTimeGetCurrent() - recoveryStarted < 5.0) {
            const size_t availableNow = trace.availableBytes();
            consecutivePasses = availableNow >= kSerialDiTRequiredBytes ? consecutivePasses + 1 : 0;
            trace.record([NSString stringWithFormat:@"Encoder 释放后内存观察：%.2f 秒；连续达标 %d/2",
                          CFAbsoluteTimeGetCurrent() - recoveryStarted, consecutivePasses]);
            if (consecutivePasses >= 2) break;
            usleep(250000);
        }
    } else {
        usleep(1000000);
        trace.record(@"Encoder 释放后 1 秒");
        usleep(4000000);
        trace.record(@"Encoder 释放后 5 秒");
    }
    const size_t available = trace.availableBytes();
    if (available < kSerialDiTRequiredBytes) {
        trace.record([NSString stringWithFormat:@"安全停止：DiT 需至少 %.0f MB 可用；当前 %.0f MB，未加载 DiT",
                      kSerialDiTRequiredBytes / 1000000.0, available / 1000000.0]);
        return failure(@"Encoder 已完成，但内存余量不足；未启动 DiT。请分享完整串行日志。");
    }
    const double recoverySeconds = CFAbsoluteTimeGetCurrent() - recoveryStarted;
    trace.record([NSString stringWithFormat:@"内存门槛通过；观察 %.3f 秒；开始 DiT 八步与 VAE %d 段（详见 %@）",
                  recoverySeconds, optimized ? 6 : 12,
                  cpuVAE ? @"last-inference-cpu-vae.log" : @"last-inference.log"]);
    NSDictionary<NSString *, id> *song = cpuVAE
        ? [FixedSongEngine generateOptimizedCpuVAEAtDirectory:directory]
        : (optimized ? [FixedSongEngine generateOptimizedAtDirectory:directory]
                     : [FixedSongEngine generateAtDirectory:directory]);
    trace.record([NSString stringWithFormat:@"DiT/VAE 已返回：%@；%@",
                  [song[@"success"] boolValue] ? @"成功" : @"失败", song[@"message"] ?: @"无消息"]);
    if (![song[@"success"] boolValue]) {
        return failure(@"Encoder 已完成，但 DiT/VAE 或 WAV 写入未通过。请分享完整串行及推理日志。");
    }
    NSString *wavPath = song[@"wavPath"];
    if (![wavPath isKindOfClass:NSString.class] || wavPath.length == 0) {
        trace.record(@"失败：推理报告成功，但没有返回 WAV 路径");
        return failure(@"推理报告成功，但没有返回 WAV 路径。请分享完整串行日志。");
    }
    const double elapsed = CFAbsoluteTimeGetCurrent() - started;
    trace.record([NSString stringWithFormat:@"%@串行完成；总耗时 %.3f 秒（内存回收观察 %.3f 秒）；30 秒音频 RTF %.3f；VAE=%@；歌词条件仍为固定样例",
                  optimized ? @"优化" : @"完整", elapsed, recoverySeconds, elapsed / 30.0,
                  cpuVAE ? @"CPU Low" : @"Metal Low 请求"]);
    return @{ @"success": @YES, @"wavPath": wavPath,
              @"elapsedSeconds": @(elapsed), @"audioSeconds": @30.0,
              @"backend": cpuVAE ? @"Encoder→DiT Metal→VAE CPU Low" : @"Encoder→DiT→VAE 串行 / Metal 请求、CPU 回退可能",
              @"message": cpuVAE
                  ? @"CPU Low VAE 对照已生成 30 秒 WAV；Encoder 输出仍未接入 DiT，歌词为固定条件。请分享 CPU 串行和推理日志并试听。"
                  : @"三组件串行推理已完成并保存 30 秒 WAV；Encoder 输出尚未接入 DiT，歌词仍为固定条件。请分享完整串行和推理日志。" };
}

+ (NSDictionary<NSString *, id> *)runSerialDiTAtDirectory:(NSString *)directory {
    Trace trace("encoder-dit-serial.log",
                @"开始：Encoder 全序列前向 → 释放 → DiT Metal High 两步；固定输入，不运行 VAE");
    NSDictionary<NSString *, id> *encoder = [self runAtDirectory:directory];
    trace.record(@"Encoder 测试函数已返回；立即采样内存");
    if (![encoder[@"success"] boolValue]) {
        trace.record([NSString stringWithFormat:@"停止：Encoder 失败；%@", encoder[@"message"] ?: @"未知原因"]);
        return encoder;
    }
    usleep(1000000);
    trace.record(@"Encoder 释放后 1 秒");
    usleep(4000000);
    trace.record(@"Encoder 释放后 5 秒");
    const size_t available = trace.availableBytes();
    if (available < kSerialDiTRequiredBytes) {
        trace.record([NSString stringWithFormat:@"安全停止：DiT 需至少 %.0f MB 可用；当前 %.0f MB，未加载 DiT",
                      kSerialDiTRequiredBytes / 1000000.0, available / 1000000.0]);
        return failure(@"Encoder 已完成；释放 5 秒后内存余量仍不足，未启动 DiT。请分享串行和 Encoder 日志。");
    }
    trace.record(@"内存门槛通过；开始 DiT Metal High 两步（详见 last-inference.log）");
    NSDictionary<NSString *, id> *dit = [FixedSongEngine probeTwoStepHighAtDirectory:directory];
    trace.record([NSString stringWithFormat:@"DiT 已返回：%@；%@",
                  [dit[@"success"] boolValue] ? @"成功" : @"失败", dit[@"message"] ?: @"无消息"]);
    if (![dit[@"success"] boolValue]) {
        return failure(@"Encoder 已完成，DiT 两步未通过。请分享串行和 DiT 日志。");
    }
    return @{ @"success": @YES,
              @"message": @"Encoder 释放后，DiT Metal High 两步串行测试完成；测试使用各自固定输入，尚未连接歌词条件。请分享串行和 DiT 日志。" };
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory {
    return [self runAtDirectory:directory warmRuns:kWarmRuns];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory warmRuns:(size_t)warmRuns {
    @autoreleasepool {
        Trace trace;
        NSURL *idsURL = [NSBundle.mainBundle URLForResource:@"encoder-input-ids" withExtension:@"i64" subdirectory:@"Probe"];
        NSData *idsData = idsURL ? [NSData dataWithContentsOfURL:idsURL] : nil;
        if (!idsData || idsData.length != kTokens * sizeof(int64_t)) {
            trace.record(@"失败：107-token 测试输入缺失");
            return failure(@"Encoder 测试输入缺失。请重新安装 App。");
        }
        NSString *path = [directory stringByAppendingPathComponent:@"text_encoder_w4.mnn"];
        NSString *weights = [directory stringByAppendingPathComponent:@"text_encoder_w4.mnn.weight"];
        if (![NSFileManager.defaultManager fileExistsAtPath:path] ||
            ![NSFileManager.defaultManager fileExistsAtPath:weights]) {
            trace.record(@"失败：Encoder 模型或外部权重缺失");
            return failure(@"Encoder 模型或外部权重缺失。");
        }
        if (!trace.hasHeadroom()) return failure(@"可用内存不足 512 MiB，未启动 Encoder。");

        Graph graph;
        const CFAbsoluteTime loadStarted = CFAbsoluteTimeGetCurrent();
        graph.interpreter = MNN::Interpreter::createFromFile(path.fileSystemRepresentation);
        trace.record([NSString stringWithFormat:@"读取 Encoder 图 %.3f 秒", CFAbsoluteTimeGetCurrent() - loadStarted]);
        if (!graph.interpreter) return failure(@"无法读取 Encoder MNN 图。");
        MNN::BackendConfig backend;
        backend.precision = MNN::BackendConfig::Precision_Low;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_METAL;
        config.backupType = MNN_FORWARD_CPU;
        config.backendConfig = &backend;
        const CFAbsoluteTime sessionStarted = CFAbsoluteTimeGetCurrent();
        graph.session = graph.interpreter->createSession(config);
        const double sessionSeconds = CFAbsoluteTimeGetCurrent() - sessionStarted;
        trace.record([NSString stringWithFormat:@"Encoder 会话创建 %.3f 秒", sessionSeconds]);
        if (!graph.session) return failure(@"无法创建 Encoder 会话；可能是内存不足或算子不支持。");
        graph.interpreter->releaseModel();
        trace.record(@"Encoder 原始模型已释放");
        if (!trace.hasHeadroom()) {
            trace.record(@"安全停止：会话创建后可用内存不足 512 MiB");
            return failure(@"Encoder 会话已创建，但可用内存不足 512 MiB，未运行前向。请分享日志。");
        }

        double coldSeconds = 0.0;
        double warmTotal = 0.0;
        for (size_t run = 0; run <= warmRuns; ++run) {
            if (!trace.hasHeadroom()) {
                trace.record(@"安全停止：可用内存不足 512 MiB");
                return failure(@"Encoder 可用内存低于 512 MiB，已停止。请分享日志。");
            }
            if (!setInput(graph, idsData)) {
                trace.record(@"失败：Encoder input_ids 类型或形状不匹配");
                return failure(@"Encoder input_ids 类型或形状不匹配。");
            }
            trace.record([NSString stringWithFormat:@"Encoder %@前向开始", run == 0 ? @"首次" : @"预热后"]);
            const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
            const MNN::ErrorCode code = graph.interpreter->runSession(graph.session);
            float peak = 0.0f;
            const bool readOK = code == MNN::NO_ERROR && readOutput(graph, peak);
            const double seconds = CFAbsoluteTimeGetCurrent() - started;
            if (!readOK) {
                trace.record([NSString stringWithFormat:@"Encoder 第 %zu 次前向失败或输出非有限值；MNN=%d", run + 1, static_cast<int>(code)]);
                return failure([NSString stringWithFormat:@"Encoder 第 %zu 次前向失败或输出非有限值。请分享日志。", run + 1]);
            }
            trace.record([NSString stringWithFormat:@"Encoder 第 %zu 次前向完成；run+read=%.3f 秒；输出峰值 %.8g", run + 1, seconds, peak]);
            if (run == 0) coldSeconds = seconds;
            else warmTotal += seconds;
        }
        const double warmSeconds = warmRuns ? warmTotal / warmRuns : 0.0;
        if (warmRuns) {
            trace.record([NSString stringWithFormat:@"完成：首次 %.3f 秒；预热后平均 %.3f 秒；107 token 全序列吞吐 %.1f token/秒；采样峰值 %.0f MB；decode=不适用（无 KV cache）",
                          coldSeconds, warmSeconds, kTokens / warmSeconds, trace.sampledPeak / 1000000.0]);
        } else {
            trace.record([NSString stringWithFormat:@"完成：一次全序列前向 %.3f 秒；采样峰值 %.0f MB；decode=不适用（无 KV cache）",
                          coldSeconds, trace.sampledPeak / 1000000.0]);
        }
        graph.interpreter->releaseSession(graph.session);
        graph.session = nullptr;
        MNN::Interpreter::destroy(graph.interpreter);
        graph.interpreter = nullptr;
        trace.record(@"Encoder 会话已释放");
        return @{ @"success": @YES,
                  @"message": @"Encoder 107-token 全序列前向已测；此图无 KV cache，不能测逐 token decode。请分享测速日志。",
                  @"sessionSeconds": @(sessionSeconds), @"coldSeconds": @(coldSeconds),
                  @"warmSeconds": @(warmSeconds), @"sampledPeakMB": @(trace.sampledPeak / 1000000.0) };
    }
}

@end
