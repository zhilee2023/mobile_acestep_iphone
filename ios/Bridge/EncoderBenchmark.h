#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface EncoderBenchmark : NSObject
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runSerialDiTAtDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END
