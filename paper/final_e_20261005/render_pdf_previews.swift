import Foundation
import AppKit
import PDFKit
import Darwin
let lock = open("/tmp/truth-probing-cpu-exclusive.lock", O_WRONLY | O_CREAT, 0o644)
guard flock(lock, LOCK_EX | LOCK_NB) == 0 else { fatalError("CPU slot occupied") }
defer { close(lock) }
let root = URL(fileURLWithPath: CommandLine.arguments[1])
let out = URL(fileURLWithPath: CommandLine.arguments[2])
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
for folder in ["figures", "tables"] {
    for file in try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent(folder), includingPropertiesForKeys: nil).filter({$0.pathExtension == "pdf"}) {
        guard let doc = PDFDocument(url: file), doc.pageCount == 1, let page = doc.page(at: 0) else { fatalError(file.path) }
        let b = page.bounds(for: .mediaBox)
        let size = NSSize(width: 1800, height: 1800 * b.height / b.width)
        let img = page.thumbnail(of: size, for: .mediaBox)
        guard let tiff = img.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff), let png = bitmap.representation(using: .png, properties: [:]) else {fatalError("render")}
        let dest = out.appendingPathComponent(folder + "-" + file.deletingPathExtension().lastPathComponent + ".png")
        try png.write(to: dest)
        print(dest.path)
    }
}
