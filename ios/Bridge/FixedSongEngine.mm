#import "FixedSongEngine.h"
#import <MNN/Interpreter.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr int kLatentFrames = 750;
constexpr int kLatentChannels = 64;
constexpr int kSampleRate = 48000;
constexpr int kUpsample = 1920;
constexpr int kSamples = kLatentFrames * kUpsample;
constexpr std::array<float, 8> kTimesteps = {1.0f, 0.875f, 0.75f, 0.625f,
                                              0.5f, 0.375f, 0.25f, 0.125f};

NSDictionary<NSString *, id> *failure(NSString *message) {
    return @{ @"success": @NO, @"message": message };
}

struct Graph {
    MNN::Interpreter *interpreter = nullptr;
    MNN::Session *session = nullptr;
    ~Graph() {
        if (interpreter && session) interpreter->releaseSession(session);
        if (interpreter) MNN::Interpreter::destroy(interpreter);
    }
};

std::unique_ptr<Graph> openGraph(NSString *path) {
    auto graph = std::make_unique<Graph>();
    graph->interpreter = MNN::Interpreter::createFromFile(path.fileSystemRepresentation);
    if (!graph->interpreter) return nullptr;
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    backend.memory = MNN::BackendConfig::Memory_Low;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_METAL;
    config.backupType = MNN_FORWARD_CPU;
    config.backendConfig = &backend;
    graph->session = graph->interpreter->createSession(config);
    if (!graph->session) return nullptr;
    return graph;
}

bool setInput(Graph &graph, const char *name, const float *values, size_t count) {
    MNN::Tensor *input = graph.interpreter->getSessionInput(graph.session, name);
    if (!input || input->usize() != count * sizeof(float)) return false;
    MNN::Tensor host(input, input->getDimensionType(), true);
    if (!host.host<float>()) return false;
    std::memcpy(host.host<float>(), values, count * sizeof(float));
    if (!input->copyFromHostTensor(&host)) {
        if (!input->host<float>()) return false;
        std::memcpy(input->host<float>(), values, count * sizeof(float));
    }
    return true;
}

bool readOutput(Graph &graph, const char *name, size_t count, std::vector<float> &values) {
    MNN::Tensor *output = graph.interpreter->getSessionOutput(graph.session, name);
    if (!output || output->usize() != count * sizeof(float)) return false;
    MNN::Tensor host(output, output->getDimensionType(), true);
    if (!host.host<float>()) return false;
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

NSURL *saveWAV(const std::vector<float> &audio) {
    float peak = 0.0f;
    for (float sample : audio) {
        if (!std::isfinite(sample)) return nil;
        peak = std::max(peak, std::abs(sample));
    }
    if (peak < 1e-6f) return nil;
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
    if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:nil]) return nil;
    NSString *filename = [NSString stringWithFormat:@"fixed-30s-w4-%@.wav", [[NSUUID UUID] UUIDString]];
    NSURL *target = [directory URLByAppendingPathComponent:filename];
    NSData *data = [NSData dataWithBytes:wave.data() length:wave.size()];
    return [data writeToURL:target atomically:YES] ? target : nil;
}
} // namespace

@implementation FixedSongEngine

+ (NSDictionary<NSString *, id> *)generateAtDirectory:(NSString *)directory {
    @autoreleasepool {
        const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        const size_t latentCount = static_cast<size_t>(kLatentFrames) * kLatentChannels;
        std::vector<float> hidden, encoder, context;
        if (!loadFixture(@"dit-hidden", latentCount, hidden) ||
            !loadFixture(@"dit-encoder", 181 * 2048, encoder) ||
            !loadFixture(@"dit-context", static_cast<size_t>(kLatentFrames) * 128, context)) {
            return failure(@"固定歌曲条件文件缺失；无法生成。");
        }

        NSString *ditPath = [directory stringByAppendingPathComponent:@"dit_w4.mnn"];
        if (![NSFileManager.defaultManager fileExistsAtPath:[directory stringByAppendingPathComponent:@"dit_w4.mnn.weight"]]) {
            return failure(@"DiT 外部权重缺失。");
        }
        {
            auto dit = openGraph(ditPath);
            if (!dit) return failure(@"无法创建 DiT MNN 会话；可能是内存不足或算子不支持。");
            if (!setInput(*dit, "encoder_hidden_states", encoder.data(), encoder.size()) ||
                !setInput(*dit, "context_latents", context.data(), context.size())) {
                return failure(@"DiT 固定条件输入形状不匹配。");
            }
            std::vector<float> velocity;
            for (size_t step = 0; step < kTimesteps.size(); ++step) {
                float time = kTimesteps[step];
                if (!setInput(*dit, "hidden_states", hidden.data(), hidden.size()) ||
                    !setInput(*dit, "timestep", &time, 1) ||
                    !setInput(*dit, "timestep_r", &time, 1)) {
                    return failure(@"DiT 采样输入形状不匹配。");
                }
                MNN::ErrorCode code = dit->interpreter->runSession(dit->session);
                if (code != MNN::NO_ERROR || !readOutput(*dit, "denoised", latentCount, velocity)) {
                    return failure([NSString stringWithFormat:@"DiT 第 %zu/8 步失败（MNN %d）。", step + 1, static_cast<int>(code)]);
                }
                float next = step + 1 < kTimesteps.size() ? kTimesteps[step + 1] : 0.0f;
                float delta = time - next;
                for (size_t index = 0; index < latentCount; ++index) hidden[index] -= velocity[index] * delta;
            }
        }

        std::vector<float> audio(static_cast<size_t>(2) * kSamples);
        int written = 0;
        for (int chunk = 0; chunk < 6; ++chunk) {
            const int coreStart = chunk * 128;
            const int coreEnd = std::min(coreStart + 128, kLatentFrames);
            const int winStart = std::max(0, coreStart - 64);
            const int winEnd = std::min(kLatentFrames, coreEnd + 64);
            const int frames = winEnd - winStart;
            NSString *vaeName = [NSString stringWithFormat:@"vae_decoder_%d_w4.mnn", frames];
            auto vae = openGraph([directory stringByAppendingPathComponent:vaeName]);
            if (!vae) return failure([NSString stringWithFormat:@"无法加载 VAE %d 帧模型。", frames]);
            std::vector<float> latent(static_cast<size_t>(kLatentChannels) * frames);
            for (int channel = 0; channel < kLatentChannels; ++channel) {
                for (int frame = winStart; frame < winEnd; ++frame) {
                    latent[channel * frames + frame - winStart] = hidden[frame * kLatentChannels + channel];
                }
            }
            if (!setInput(*vae, "latents", latent.data(), latent.size())) {
                return failure([NSString stringWithFormat:@"VAE %d 帧输入不匹配。", frames]);
            }
            MNN::ErrorCode code = vae->interpreter->runSession(vae->session);
            std::vector<float> decoded;
            if (code != MNN::NO_ERROR || !readOutput(*vae, "waveform", static_cast<size_t>(2) * frames * kUpsample, decoded)) {
                return failure([NSString stringWithFormat:@"VAE 第 %d/6 段失败（MNN %d）。", chunk + 1, static_cast<int>(code)]);
            }
            const int coreSamples = (coreEnd - coreStart) * kUpsample;
            const int offset = (coreStart - winStart) * kUpsample;
            for (int channel = 0; channel < 2; ++channel) {
                const float *source = decoded.data() + channel * frames * kUpsample + offset;
                float *target = audio.data() + channel * kSamples + written;
                std::memcpy(target, source, static_cast<size_t>(coreSamples) * sizeof(float));
            }
            written += coreSamples;
        }
        if (written != kSamples) return failure(@"VAE 拼接后的音频长度不等于 30 秒。");
        NSURL *target = saveWAV(audio);
        if (!target) return failure(@"音频无效或写入 WAV 失败。");
        return @{ @"success": @YES, @"message": @"iPhone 已完成 DiT 八步及 VAE 六段推理。歌词条件为预计算固定样例。",
                  @"wavPath": target.path, @"elapsedSeconds": @(CFAbsoluteTimeGetCurrent() - started),
                  @"audioSeconds": @30.0, @"backend": @"MNN Metal 请求 / CPU 回退可能" };
    }
}

@end
