#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface AceBridge : NSObject
+ (NSDictionary<NSString *, NSNumber *> *)modelStatusAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runDiTProbeAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runTwoStepHighProbeAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runTwoStepLowProbeAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runEncoderBenchmarkAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runEncoderDiTSerialProbeAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateFullSerialSongAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateOptimizedSerialSongAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateCpuVaeSerialSongAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)diagnoseVAEAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)diagnoseVAENoTuneAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateFixedSongAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateCustomLyrics:(NSString *)lyrics atDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END

