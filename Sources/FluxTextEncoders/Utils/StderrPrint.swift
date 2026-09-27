// StderrPrint.swift - Module-scope stderr shadow for `print`
// Copyright 2025 Vincent Gourbin

import Foundation

// Why: stdout is reserved for machine-readable data. In particular,
// `vinetas generate -o -` streams a raw PNG on stdout; any diagnostic or
// progress text that lands on stdout corrupts that byte stream. This
// module's ~109 bare `print(` call sites (plus every call routed through
// `FluxDebug.log`/`FluxDebug.error`) were written against `Swift.print`,
// which writes to stdout by default.
//
// How: Swift resolves an unqualified `print` call to the innermost scope
// that declares one. A module-scope function named `print` therefore
// shadows `Swift.print` for every unqualified call in this module, with
// zero call-site edits required. This function reproduces `Swift.print`'s
// exact formatting (items joined by `separator`, followed by `terminator`)
// but writes the resulting bytes to `FileHandle.standardError` instead of
// standard output. Call `Swift.print(...)` explicitly at any site that
// must still emit machine-readable data on stdout.
func print(_ items: Any..., separator: String = " ", terminator: String = "\n") {
  let message = items.map { "\($0)" }.joined(separator: separator) + terminator
  if let data = message.data(using: .utf8) {
    FileHandle.standardError.write(data)
  }
}
