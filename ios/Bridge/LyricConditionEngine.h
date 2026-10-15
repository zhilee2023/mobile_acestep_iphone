#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LyricConditionEngine : NSObject
+ (NSDictionary<NSString *, id> *)encodeLyrics:(NSString *)lyrics atDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END
