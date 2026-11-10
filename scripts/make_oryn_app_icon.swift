// Renders the ORYN EDGE app icon: ORYN's mark in bone on ink.
//
//   swift scripts/make_oryn_app_icon.swift <oryn-mark.svg> <out.png>
//
// The mark is ORYN's own vector artwork (ORYN-WEBSITE/public/brand/
// oryn-mark-light.svg), rasterised as-is — never redrawn — and centred on its
// artwork bounds. Same output contract as make_app_icon.swift: exactly
// 1024×1024 pixels, no alpha channel (App Store review requires both), drawn
// into a CGContext so a Retina display cannot double the pixel size.
import AppKit
import CoreGraphics
import UniformTypeIdentifiers

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    FileHandle.standardError.write("usage: make_oryn_app_icon.swift <mark.svg> <out.png>\n".data(using: .utf8)!)
    exit(2)
}
let svgURL = URL(fileURLWithPath: arguments[1])
let outURL = URL(fileURLWithPath: arguments[2])

guard let mark = NSImage(contentsOf: svgURL), mark.size.width > 0 else {
    fatalError("could not load \(svgURL.path) as an image")
}

let pixels = 1024
let size = CGFloat(pixels)
guard let context = CGContext(
    data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
) else { fatalError("could not create context") }

// Field: ORYN ink-900 (#0A0B0C).
context.setFillColor(CGColor(red: 0x0A / 255.0, green: 0x0B / 255.0, blue: 0x0C / 255.0, alpha: 1))
context.fill(CGRect(x: 0, y: 0, width: size, height: size))

// The mark at 46% of the tile height — inside the icon grid's safe circle with
// room to breathe, and true to the artwork's own aspect ratio (117.23 × 144.33).
let markHeight = size * 0.46
let markWidth = markHeight * (mark.size.width / mark.size.height)
let rect = CGRect(x: (size - markWidth) / 2, y: (size - markHeight) / 2, width: markWidth, height: markHeight)

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
mark.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1)
NSGraphicsContext.restoreGraphicsState()

guard let image = context.makeImage(),
      let destination = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.png.identifier as CFString, 1, nil)
else { fatalError("could not encode PNG") }
CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else { fatalError("could not write \(outURL.path)") }
print("wrote \(outURL.path) (\(image.width)×\(image.height))")
