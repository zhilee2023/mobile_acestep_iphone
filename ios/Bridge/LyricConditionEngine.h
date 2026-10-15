#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LyricConditionEngine : NSObject
+ (NSDictionary<NSString *, id> *)encodeLyrics:(NSString *)lyrics atDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)tokenizeCaption:(NSString *)caption;
@end

NS_ASSUME_NONNULL_END
