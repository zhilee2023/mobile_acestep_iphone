#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface FixedSongEngine : NSObject
+ (NSDictionary<NSString *, id> *)generateAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateOptimizedAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateOptimizedCpuVAEAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateOptimizedNoTuneVAEAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateWithCondition:(NSData *)condition atDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateWithConditionCpuVAE:(NSData *)condition atDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateRepaintTrialAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)diagnoseVAEAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)diagnoseVAENoTuneAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)probeTwoStepHighAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)probeTwoStepLowAtDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END
