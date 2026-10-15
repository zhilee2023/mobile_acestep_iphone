import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let tokenizerURL = root.appendingPathComponent("ios/Tokenizer/tokenizer.json")
let captionURL = root.appendingPathComponent("ios/Tokenizer/sample-caption.txt")
let referenceURL = root.appendingPathComponent("ios/Probe/encoder-input-ids.i64")
let json = try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerURL)) as! [String: Any]
let model = json["model"] as! [String: Any]
let vocab = model["vocab"] as! [String: Int]
let merges = model["merges"] as! [[String]]
let sequence = json["pre_tokenizer"] as! [String: Any]
let parts = sequence["pretokenizers"] as! [[String: Any]]
let split = parts[0]["pattern"] as! [String: String]
let regex = try NSRegularExpression(pattern: split["Regex"]!)
var ranks: [String: Int] = [:]
for (index, pair) in merges.enumerated() where pair.count == 2 {
    ranks[pair[0] + "\u{1}" + pair[1]] = index
}
var base = Array(33...126) + Array(161...172) + Array(174...255)
var codes = base
var added = 0
for value in 0...255 where !base.contains(value) {
    base.append(value)
    codes.append(256 + added)
    added += 1
}
var symbols = Array(repeating: "", count: 256)
for index in base.indices {
    symbols[base[index]] = String(UnicodeScalar(codes[index])!)
}
let caption = try String(contentsOf: captionURL, encoding: .utf8)
    .trimmingCharacters(in: .newlines)
let raw = "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\n\(caption)\n\n# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n"
let normalized = raw.precomposedStringWithCanonicalMapping
var ids: [Int] = []
var origin: [String] = []
var covered = 0
let matches = regex.matches(in: normalized, range: NSRange(location: 0, length: (normalized as NSString).length))
for match in matches {
    precondition(match.range.location == covered, "regex gap at \(covered)")
    covered = NSMaxRange(match.range)
    let piece = (normalized as NSString).substring(with: match.range)
    var tokens = piece.utf8.map { symbols[Int($0)] }
    while tokens.count > 1 {
        var best = Int.max
        var position = -1
        for index in 0..<(tokens.count - 1) {
            if let rank = ranks[tokens[index] + "\u{1}" + tokens[index + 1]], rank < best {
                best = rank
                position = index
            }
        }
        if position == -1 { break }
        tokens[position] += tokens[position + 1]
        tokens.remove(at: position + 1)
    }
    for token in tokens {
        ids.append(vocab[token]!)
        origin.append(piece)
    }
}
precondition(covered == (normalized as NSString).length)
ids += [151643, 198, 151643]
origin += ["EOT", "newline", "EOT"]
let reference = try Data(contentsOf: referenceURL)
let bytes = Array(reference)
var expected: [Int] = []
for offset in stride(from: 0, to: bytes.count, by: 8) {
    var value: UInt64 = 0
    for index in 0..<8 { value |= UInt64(bytes[offset + index]) << (8 * index) }
    expected.append(Int(value))
}
print("Apple Foundation regex matches=\(matches.count), raw tokens=\(ids.count - 3), total=\(ids.count), reference=\(expected.count)")
if let first = zip(ids, expected).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
    let index = first.offset
    print("First mismatch index=\(index), actual=\(ids[index]), expected=\(expected[index]), piece=\(String(reflecting: origin[index]))")
    let lower = max(0, index - 4)
    let upper = min(ids.count, index + 8)
    for i in lower..<upper { print("  \(i): actual=\(ids[i]) expected=\(i < expected.count ? expected[i] : -1) piece=\(String(reflecting: origin[i]))") }
} else {
    print(ids == expected ? "Exact match" : "Matching prefix, different length")
}
