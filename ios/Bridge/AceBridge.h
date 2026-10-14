#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface AceBridge : NSObject
+ (NSDictionary<NSString *, NSNumber *> *)modelStatusAtDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END

