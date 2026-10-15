#import "LyricConditionEngine.h"
#import <MNN/Interpreter.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr int kLyricTokens = 73;
constexpr int kEmbeddingWidth = 1024;
constexpr int kConditionWidth = 2048;
constexpr int kConditionTokens = 181;
constexpr int kVocabSize = 151669;
constexpr int kEndOfText = 151643;
constexpr size_t kEmbeddingBytes = static_cast<size_t>(kVocabSize) * kEmbeddingWidth * 2;

NSDictionary<NSString *, id> *failure(NSString *message, NSInteger tokenCount = 0) {
    return @{ @"success": @NO, @"message": message, @"tokenCount": @(tokenCount) };
}

NSString *pairKey(NSString *left, NSString *right) {
    return [NSString stringWithFormat:@"%@\u0001%@", left, right];
}

NSArray<NSString *> *byteSymbols() {
    NSMutableArray<NSNumber *> *base = [NSMutableArray array];
    for (int value = 33; value <= 126; ++value) [base addObject:@(value)];
    for (int value = 161; value <= 172; ++value) [base addObject:@(value)];
    for (int value = 174; value <= 255; ++value) [base addObject:@(value)];
    NSMutableArray<NSNumber *> *codes = [base mutableCopy];
    int added = 0;
    for (int value = 0; value < 256; ++value) {
        if (![base containsObject:@(value)]) {
            [base addObject:@(value)];
            [codes addObject:@(256 + added++)];
        }
    }
    NSMutableArray<NSString *> *symbols = [NSMutableArray arrayWithCapacity:256];
    for (int value = 0; value < 256; ++value) [symbols addObject:@""];
    for (int index = 0; index < base.count; ++index) {
        unichar character = static_cast<unichar>(codes[index].intValue);
        symbols[base[index].intValue] = [NSString stringWithCharacters:&character length:1];
    }
    return symbols;
}

NSArray<NSNumber *> *tokenize(NSString *lyrics, NSString **error) {
    if (lyrics.length == 0 || [lyrics containsString:@"<|"]) {
        if (error) *error = @"歌词为空或包含模型保留标记 <|，请修改后重试。";
        return nil;
    }
    NSURL *url = [NSBundle.mainBundle URLForResource:@"tokenizer" withExtension:@"json" subdirectory:@"Tokenizer"];
    NSData *data = url ? [NSData dataWithContentsOfURL:url] : nil;
    NSDictionary *root = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    NSDictionary *model = root[@"model"];
    NSDictionary<NSString *, NSNumber *> *vocab = model[@"vocab"];
    NSArray<NSArray<NSString *> *> *merges = model[@"merges"];
    NSArray *pretokenizers = root[@"pre_tokenizer"][@"pretokenizers"];
    NSString *pattern = pretokenizers.firstObject[@"pattern"][@"Regex"];
    if (![vocab isKindOfClass:NSDictionary.class] || ![merges isKindOfClass:NSArray.class] || !pattern) {
        if (error) *error = @"内置分词表不可用。";
        return nil;
    }
    NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
    if (!regex) {
        if (error) *error = @"iPhone 无法编译歌词分词规则。";
        return nil;
    }
    NSMutableDictionary<NSString *, NSNumber *> *ranks = [NSMutableDictionary dictionaryWithCapacity:merges.count];
    for (NSUInteger rank = 0; rank < merges.count; ++rank) {
        NSArray<NSString *> *pair = merges[rank];
        if (pair.count == 2) ranks[pairKey(pair[0], pair[1])] = @(rank);
    }
    NSArray<NSString *> *symbols = byteSymbols();
    NSString *normalized = [[NSString stringWithFormat:@"# Languages\nzh\n\n# Lyric\n%@", lyrics]
                            precomposedStringWithCanonicalMapping];
    NSArray<NSTextCheckingResult *> *matches = [regex matchesInString:normalized options:0
                                                                range:NSMakeRange(0, normalized.length)];
    NSMutableArray<NSNumber *> *ids = [NSMutableArray array];
    NSUInteger covered = 0;
    for (NSTextCheckingResult *match in matches) {
        if (match.range.location != covered) {
            if (error) *error = @"歌词含有当前分词器无法识别的字符。";
            return nil;
        }
        covered = NSMaxRange(match.range);
        NSString *piece = [normalized substringWithRange:match.range];
        NSData *utf8 = [piece dataUsingEncoding:NSUTF8StringEncoding];
        const uint8_t *bytes = static_cast<const uint8_t *>(utf8.bytes);
        NSMutableArray<NSString *> *tokens = [NSMutableArray arrayWithCapacity:utf8.length];
        for (NSUInteger index = 0; index < utf8.length; ++index) [tokens addObject:symbols[bytes[index]]];
        while (tokens.count > 1) {
            NSUInteger best = NSUIntegerMax;
            NSUInteger bestIndex = NSNotFound;
            for (NSUInteger index = 0; index + 1 < tokens.count; ++index) {
                NSNumber *rank = ranks[pairKey(tokens[index], tokens[index + 1])];
                if (rank && rank.unsignedIntegerValue < best) {
                    best = rank.unsignedIntegerValue;
                    bestIndex = index;
                }
            }
            if (bestIndex == NSNotFound) break;
            tokens[bestIndex] = [tokens[bestIndex] stringByAppendingString:tokens[bestIndex + 1]];
            [tokens removeObjectAtIndex:bestIndex + 1];
        }
        for (NSString *token in tokens) {
            NSNumber *identifier = vocab[token];
            if (!identifier) {
                if (error) *error = @"歌词分词表缺少字节组合。";
                return nil;
            }
            [ids addObject:identifier];
        }
        if (ids.count > kLyricTokens) break;
    }
    if (covered != normalized.length && ids.count <= kLyricTokens) {
        if (error) *error = @"歌词未能完整分词。";
        return nil;
    }
    [ids addObject:@(kEndOfText)];
    [ids addObject:@(kEndOfText)];
    return ids;
}

bool readProbe(NSString *name, NSString *extension, size_t bytes, std::vector<uint8_t> &result) {
    NSURL *url = [NSBundle.mainBundle URLForResource:name withExtension:extension subdirectory:@"Probe"];
    NSData *data = url ? [NSData dataWithContentsOfURL:url] : nil;
    if (!data || data.length != bytes) return false;
    result.assign(static_cast<const uint8_t *>(data.bytes),
                  static_cast<const uint8_t *>(data.bytes) + bytes);
    return true;
}
} // namespace

@implementation LyricConditionEngine

+ (NSDictionary<NSString *, id> *)encodeLyrics:(NSString *)lyrics atDirectory:(NSString *)directory {
    @autoreleasepool {
        NSString *tokenError = nil;
        NSArray<NSNumber *> *ids = tokenize(lyrics, &tokenError);
        if (!ids) return failure(tokenError ?: @"歌词分词失败。");
        if (ids.count != kLyricTokens) {
            return failure([NSString stringWithFormat:@"当前试验版只支持恰好 %d 个 token；这段歌词为 %lu 个。请调整字数或标点后重试，不会截断。",
                            kLyricTokens, static_cast<unsigned long>(ids.count)], ids.count);
        }
        std::vector<uint8_t> sampleIds;
        if (!readProbe(@"lyric-sample-ids", @"i32", kLyricTokens * sizeof(int32_t), sampleIds)) {
            return failure(@"内置分词校验文件缺失。", ids.count);
        }
        NSURL *sampleURL = [NSBundle.mainBundle URLForResource:@"sample-lyrics" withExtension:@"txt" subdirectory:@"Tokenizer"];
        NSString *sample = sampleURL ? [NSString stringWithContentsOfURL:sampleURL encoding:NSUTF8StringEncoding error:nil] : nil;
        if (!sample) return failure(@"内置歌词样例缺失。", ids.count);
        // Run the exact upstream tokenizer contract before accepting any custom input.
        NSString *sampleError = nil;
        NSArray<NSNumber *> *sampleTokens = tokenize([sample stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]], &sampleError);
        if (!sampleTokens || sampleTokens.count != kLyricTokens) {
            return failure([NSString stringWithFormat:@"分词器自检失败：%@", sampleError ?: @"样例长度不一致"], ids.count);
        }
        const int32_t *expectedIds = reinterpret_cast<const int32_t *>(sampleIds.data());
        for (int index = 0; index < kLyricTokens; ++index) {
            if (sampleTokens[index].intValue != expectedIds[index]) {
                return failure([NSString stringWithFormat:@"分词器自检失败：样例第 %d 个 token 不一致。", index], ids.count);
            }
        }
        bool isReferenceLyrics = true;
        for (int index = 0; index < kLyricTokens; ++index) {
            if (ids[index].intValue != expectedIds[index]) {
                isReferenceLyrics = false;
                break;
            }
        }

        NSString *tablePath = [directory stringByAppendingPathComponent:@"lyric_embedding.f16"];
        NSString *graphPath = [directory stringByAppendingPathComponent:@"lyric_encoder_73_w4.mnn"];
        NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:tablePath error:nil];
        if ([attributes[NSFileSize] unsignedLongLongValue] != kEmbeddingBytes ||
            ![NSFileManager.defaultManager fileExistsAtPath:graphPath]) {
            return failure(@"歌词模型缺失或大小不符；请再导入两份歌词模型文件。", ids.count);
        }
        NSData *table = [NSData dataWithContentsOfFile:tablePath options:NSDataReadingMappedIfSafe error:nil];
        if (!table || table.length != kEmbeddingBytes) return failure(@"无法映射歌词嵌入表。", ids.count);
        const __fp16 *weights = static_cast<const __fp16 *>(table.bytes);
        std::vector<float> embeddings(static_cast<size_t>(kLyricTokens) * kEmbeddingWidth);
        for (int token = 0; token < kLyricTokens; ++token) {
            const int identifier = ids[token].intValue;
            if (identifier < 0 || identifier >= kVocabSize) return failure(@"歌词 token 越界。", ids.count);
            for (int dim = 0; dim < kEmbeddingWidth; ++dim) {
                embeddings[token * kEmbeddingWidth + dim] =
                    static_cast<float>(weights[static_cast<size_t>(identifier) * kEmbeddingWidth + dim]);
            }
        }
        table = nil;
        auto net = std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)>(
            MNN::Interpreter::createFromFile(graphPath.fileSystemRepresentation), &MNN::Interpreter::destroy);
        if (!net) return failure(@"无法读取歌词编码器模型。", ids.count);
        MNN::BackendConfig backend;
        backend.precision = MNN::BackendConfig::Precision_High;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_METAL;
        config.backupType = MNN_FORWARD_CPU;
        config.backendConfig = &backend;
        MNN::Session *session = net->createSession(config);
        if (!session) return failure(@"歌词编码器创建会话失败。", ids.count);
        net->releaseModel();
        MNN::Tensor *input = net->getSessionInput(session, "lyric_embeddings");
        if (!input) { net->releaseSession(session); return failure(@"歌词编码器输入不匹配。", ids.count); }
        MNN::Tensor host(input, input->getDimensionType(), true);
        if (!host.host<float>() || host.usize() != embeddings.size() * sizeof(float)) {
            net->releaseSession(session);
            return failure(@"歌词嵌入输入形状不匹配。", ids.count);
        }
        std::memcpy(host.host<float>(), embeddings.data(), host.usize());
        if (!input->copyFromHostTensor(&host)) {
            net->releaseSession(session);
            return failure(@"无法传入歌词嵌入。", ids.count);
        }
        MNN::ErrorCode code = net->runSession(session);
        MNN::Tensor *output = code == MNN::NO_ERROR ? net->getSessionOutput(session, "lyric_hidden_states") : nullptr;
        if (!output) { net->releaseSession(session); return failure(@"歌词编码器前向失败。", ids.count); }
        MNN::Tensor hostOutput(output, output->getDimensionType(), true);
        constexpr size_t lyricBytes = static_cast<size_t>(kLyricTokens) * kConditionWidth * sizeof(float);
        if (!hostOutput.host<float>() || hostOutput.usize() != lyricBytes || !output->copyToHostTensor(&hostOutput)) {
            net->releaseSession(session);
            return failure(@"无法读取歌词编码结果。", ids.count);
        }
        std::vector<uint8_t> combined;
        if (!readProbe(@"dit-encoder", @"f32", static_cast<size_t>(kConditionTokens) * kConditionWidth * sizeof(float), combined)) {
            net->releaseSession(session);
            return failure(@"固定风格条件缺失。", ids.count);
        }
        const float *values = hostOutput.host<float>();
        for (size_t index = 0; index < lyricBytes / sizeof(float); ++index) {
            if (!std::isfinite(values[index])) {
                net->releaseSession(session);
                return failure(@"歌词编码出现非有限值。", ids.count);
            }
        }
        if (isReferenceLyrics) {
            const float *reference = reinterpret_cast<const float *>(combined.data());
            double error = 0.0;
            double energy = 0.0;
            for (size_t index = 0; index < lyricBytes / sizeof(float); ++index) {
                const double diff = static_cast<double>(values[index]) - reference[index];
                error += diff * diff;
                energy += static_cast<double>(reference[index]) * reference[index];
            }
            const double relativeL2 = std::sqrt(error / std::max(energy, 1e-20));
            if (relativeL2 > 0.3) {
                net->releaseSession(session);
                return failure([NSString stringWithFormat:@"手机歌词编码数值对照失败（L2 %.3f）；未启动 DiT。", relativeL2], ids.count);
            }
        }
        std::memcpy(combined.data(), values, lyricBytes);
        net->releaseSession(session);
        return @{ @"success": @YES, @"tokenCount": @(ids.count),
                  @"condition": [NSData dataWithBytes:combined.data() length:combined.size()],
                  @"message": @"歌词条件编码完成。" };
    }
}
@end
