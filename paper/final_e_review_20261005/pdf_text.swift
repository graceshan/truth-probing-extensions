import Foundation
import PDFKit
import Darwin
let lock = open("/tmp/truth-probing-cpu-exclusive.lock", O_WRONLY | O_CREAT, 0o644)
guard flock(lock, LOCK_EX | LOCK_NB) == 0 else { fatalError("CPU slot occupied") }
defer { close(lock) }
let root = URL(fileURLWithPath: CommandLine.arguments[1])
var records: [String: Any] = [:]
for folder in ["figures", "tables"] {
    for file in try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent(folder), includingPropertiesForKeys: nil).filter({$0.pathExtension == "pdf"}) {
        guard let doc = PDFDocument(url: file), doc.pageCount == 1, let page = doc.page(at: 0), let text = page.string else { fatalError(file.path) }
        let b = page.bounds(for: .mediaBox)
        records[folder + "/" + file.lastPathComponent] = ["pages": doc.pageCount, "width_points": b.width, "height_points": b.height, "text": text]
    }
}
let data = try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
