#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface FixedSongEngine : NSObject
+ (NSDictionary<NSString *, id> *)generateAtDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END
