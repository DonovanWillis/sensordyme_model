// Renders the SensorDyme app icon: black field, white "S".
//
// Uses a CoreGraphics context with `noneSkipLast` rather than NSImage
// lockFocus, for two reasons found the hard way:
//   * lockFocus adopts the display's backing scale, silently emitting a 2048px
//     icon on a Retina Mac; and
//   * a 3-sample no-alpha NSBitmapImageRep is not drawable at all
//     (NSGraphicsContext(bitmapImageRep:) returns nil), while noneSkipLast is
//     both drawable and encodes to PNG with no alpha channel — which App Store
//     review requires of app icons.
import AppKit
import CoreGraphics
import CoreText
import UniformTypeIdentifiers

let pixels = 1024
let size = CGFloat(pixels)

guard let context = CGContext(
    data: nil,
    width: pixels, height: pixels,
    bitsPerComponent: 8,
    bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
) else { fatalError("could not create context") }

// Field
context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
context.fill(CGRect(x: 0, y: 0, width: size, height: size))

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)

NSGraphicsContext.restoreGraphicsState()

// Letterform, matching the in-app mark: SF Rounded Black, sheared into italic.
// Drawn as a glyph path rather than as text so it can be centred on its actual
// INK. A typographic box includes ascender and descender space the eye never
// sees, and centring on it leaves the S visibly high in the tile.
let pointSize = size * 0.74
let baseFont = NSFont(name: "SFProRounded-Black", size: pointSize)
    ?? NSFont.systemFont(ofSize: pointSize, weight: .black)

let glyph = CTFontGetGlyphWithName(baseFont as CTFont, "S" as CFString)
guard glyph != 0, let upright = CTFontCreatePathForGlyph(baseFont as CTFont, glyph, nil) else {
    fatalError("could not build glyph path")
}

// SF Rounded ships no italic face, so the slant is sheared onto the path. 12°
// is close to Apple's own optical italic and stays legible at home-screen size.
let slant = CGFloat(12.0 * .pi / 180.0)
var shear = CGAffineTransform(a: 1, b: 0, c: tan(slant), d: 1, tx: 0, ty: 0)
guard let slanted = upright.copy(using: &shear) else { fatalError("could not slant glyph") }

// Centre on the ink box, then nudge up a hair: a shape with a flat-ish bottom
// reads as sinking when it is mathematically centred.
let ink = slanted.boundingBoxOfPath
var transform = CGAffineTransform(
    translationX: (size - ink.width) / 2 - ink.minX,
    y: (size - ink.height) / 2 - ink.minY + size * 0.012
)
guard let centred = slanted.copy(using: &transform) else { fatalError("could not place glyph") }

context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
context.addPath(centred)
context.fillPath()

guard let image = context.makeImage() else { fatalError("could not snapshot context") }
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon_1024.png"
guard let destination = CGImageDestinationCreateWithURL(
    URL(fileURLWithPath: out) as CFURL, UTType.png.identifier as CFString, 1, nil
) else { fatalError("could not create png destination") }
CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else { fatalError("could not write png") }
print("wrote \(out): \(image.width)x\(image.height), alphaInfo=\(image.alphaInfo.rawValue)")
