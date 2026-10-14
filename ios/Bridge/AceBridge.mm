#import "AceBridge.h"

@implementation AceBridge

+ (NSDictionary<NSString *, NSNumber *> *)modelStatusAtDirectory:(NSString *)directory {
    NSFileManager *manager = NSFileManager.defaultManager;
    NSMutableDictionary<NSString *, NSNumber *> *status = [NSMutableDictionary dictionary];
    for (NSString *name in @[@"text_encoder_w4.mnn", @"text_encoder_w4.mnn.weight",
                             @"dit_w4.mnn", @"dit_w4.mnn.weight",
                             @"vae_decoder_192_w4.mnn", @"vae_decoder_256_w4.mnn",
                             @"vae_decoder_174_w4.mnn"]) {
        NSString *path = [directory stringByAppendingPathComponent:name];
        NSDictionary *attributes = [manager attributesOfItemAtPath:path error:nil];
        status[name] = @([attributes[NSFileSize] unsignedLongLongValue] > 0);
    }
    return status;
}

@end

