#import "EncoderBenchmark.h"
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

NSDictionary<NSString *, id> *failure(NSString *message) {
    return @{ @"success": @NO, @"message": message };
}

struct Trace {
    FILE *file = nullptr;
    uint64_t sampledPeak = 0;

    Trace() {
        NSURL *documents = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory
                                                                 inDomains:NSUserDomainMask].firstObject;
        NSURL *directory = [documents URLByAppendingPathComponent:@"Generated" isDirectory:YES];
        if (![NSFileManager.defaultManager createDirectoryAtURL:directory
                                    withIntermediateDirectories:YES attributes:nil error:nil]) return;
        NSURL *url = [directory URLByAppendingPathComponent:@"encoder-benchmark.log"];
        file = std::fopen(url.fileSystemRepresentation, "w");
        record(@"开始：Qwen3-Embedding W4；107-token 固定输入；Metal Low／低内存；无 KV-cache decode");
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

@implementation EncoderBenchmark

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory {
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
        for (size_t run = 0; run <= kWarmRuns; ++run) {
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
        const double warmSeconds = warmTotal / kWarmRuns;
        trace.record([NSString stringWithFormat:@"完成：首次 %.3f 秒；预热后平均 %.3f 秒；107 token 全序列吞吐 %.1f token/秒；采样峰值 %.0f MB；decode=不适用（无 KV cache）",
                      coldSeconds, warmSeconds, kTokens / warmSeconds, trace.sampledPeak / 1000000.0]);
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
