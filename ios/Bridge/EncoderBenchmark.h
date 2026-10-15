#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface EncoderBenchmark : NSObject
+ (NSDictionary<NSString *, id> *)encodeStyleIDs:(NSData *)idsData atDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runSerialDiTAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runFullSerialSongAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runOptimizedFullSerialAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runOptimizedCpuVaeSerialAtDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END
