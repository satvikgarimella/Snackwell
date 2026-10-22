// Encode local PNG frames using macOS AVFoundation. No network dependencies.
import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
if args.count != 6 { fatalError("Usage: encode_video FRAME_DIR OUTPUT.mp4 WIDTH HEIGHT FPS") }
let dir=URL(fileURLWithPath:args[1]), out=URL(fileURLWithPath:args[2])
let width=Int(args[3])!, height=Int(args[4])!, fps=Int32(args[5])!
let fm=FileManager.default
let frames=try fm.contentsOfDirectory(at:dir,includingPropertiesForKeys:nil)
    .filter{$0.pathExtension == "png"}.sorted{$0.lastPathComponent < $1.lastPathComponent}
if frames.isEmpty { fatalError("No frames") }
if fm.fileExists(atPath:out.path) { try fm.removeItem(at:out) }
let writer=try AVAssetWriter(outputURL:out,fileType:.mp4)
let input=AVAssetWriterInput(mediaType:.video, outputSettings:[
    AVVideoCodecKey:AVVideoCodecType.h264,
    AVVideoWidthKey:width, AVVideoHeightKey:height,
    AVVideoCompressionPropertiesKey:[AVVideoAverageBitRateKey:6000000,
                                    AVVideoProfileLevelKey:AVVideoProfileLevelH264HighAutoLevel]
])
input.expectsMediaDataInRealTime=false
let attrs:[String:Any]=[
    kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32ARGB,
    kCVPixelBufferWidthKey as String:width, kCVPixelBufferHeightKey as String:height,
    kCVPixelBufferCGImageCompatibilityKey as String:true,
    kCVPixelBufferCGBitmapContextCompatibilityKey as String:true
]
let adapter=AVAssetWriterInputPixelBufferAdaptor(assetWriterInput:input,sourcePixelBufferAttributes:attrs)
writer.add(input)
if !writer.startWriting(){ fatalError("start: \(String(describing:writer.error))") }
writer.startSession(atSourceTime:.zero)
let colors=CGColorSpaceCreateDeviceRGB()
for (i,file) in frames.enumerated(){
    autoreleasepool {
        while !input.isReadyForMoreMediaData {
            if writer.status == .failed { fatalError("\(String(describing:writer.error))") }
            Thread.sleep(forTimeInterval:0.002)
        }
        guard let src=CGImageSourceCreateWithURL(file as CFURL,nil),
              let img=CGImageSourceCreateImageAtIndex(src,0,nil) else { fatalError("Read image") }
        var maybeBuffer:CVPixelBuffer?
        let status=CVPixelBufferPoolCreatePixelBuffer(nil,adapter.pixelBufferPool!,&maybeBuffer)
        guard status == kCVReturnSuccess, let buffer=maybeBuffer else {fatalError("Pixel buffer")}
        CVPixelBufferLockBaseAddress(buffer,[])
        guard let context=CGContext(data:CVPixelBufferGetBaseAddress(buffer),width:width,height:height,
            bitsPerComponent:8,bytesPerRow:CVPixelBufferGetBytesPerRow(buffer),space:colors,
            bitmapInfo:CGImageAlphaInfo.noneSkipFirst.rawValue) else {fatalError("Context")}
        context.draw(img,in:CGRect(x:0,y:0,width:width,height:height))
        CVPixelBufferUnlockBaseAddress(buffer,[])
        if !adapter.append(buffer,withPresentationTime:CMTime(value:Int64(i),timescale:fps)) {
            fatalError("Append: \(String(describing:writer.error))")
        }
    }
}
input.markAsFinished()
let semaphore=DispatchSemaphore(value:0)
writer.finishWriting { semaphore.signal() }
semaphore.wait()
if writer.status != .completed {fatalError("Finish: \(String(describing:writer.error))")}
print("Encoded \(frames.count) frames, \(width)x\(height), \(fps) fps: \(out.path)")
// Decode a frame back out to verify video orientation and rendering.
let asset=AVURLAsset(url:out)
let generator=AVAssetImageGenerator(asset:asset)
generator.appliesPreferredTrackTransform=true
let image=try generator.copyCGImage(at:CMTime(value:5,timescale:1),actualTime:nil)
let check=out.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("source/qa/video-decoded-check.png")
if let dest=CGImageDestinationCreateWithURL(check as CFURL,UTType.png.identifier as CFString,1,nil){
    CGImageDestinationAddImage(dest,image,nil);CGImageDestinationFinalize(dest)
}
