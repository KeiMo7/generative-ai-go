// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements a single, shared pipeline for assembling and
// validating multimodal input. Every entry point that builds a request
// (single-shot and streaming generation, chat sessions, token counting,
// embeddings and cached-content creation) routes its contents through
// validateMultimodalInput, so inline bytes and file-handle references are
// recognized, metered and rejected with the same rules.
//
// The pipeline never inspects or logs the raw bytes beyond what the standard
// library needs to sniff a MIME type: observations and errors carry sizes and
// type names only, never file contents.

package genai

import (
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"sync"
)

// InputModality identifies the kind of media an input entry belongs to.
type InputModality string

const (
	// ModalityText is plain text.
	ModalityText InputModality = "text"
	// ModalityImage is a still image.
	ModalityImage InputModality = "image"
	// ModalityAudio is sound.
	ModalityAudio InputModality = "audio"
	// ModalityVideo is moving imagery.
	ModalityVideo InputModality = "video"
	// ModalityDocument is a document such as a PDF or office document.
	ModalityDocument InputModality = "document"
	// ModalityUnknown is used when a modality cannot be determined.
	ModalityUnknown InputModality = "unknown"
)

// InputSource identifies how an entry was supplied.
type InputSource string

const (
	// SourceInline means the entry was passed as inline bytes ([Blob]).
	SourceInline InputSource = "inline"
	// SourceReference means the entry was passed as a file-handle
	// reference ([FileData]).
	SourceReference InputSource = "reference"
)

// InputErrorKind is the mutually exclusive category of an input error.
type InputErrorKind string

const (
	// InputErrorMalformed: the entry is written illegally (empty or
	// malformed handle, local path, external link, syntactically invalid
	// MIME type, nil part).
	InputErrorMalformed InputErrorKind = "malformed"
	// InputErrorUnsupported: the entry is well-formed and recognizable,
	// but its type is not accepted (for example an executable or archive).
	InputErrorUnsupported InputErrorKind = "unsupported"
	// InputErrorMismatch: a declared MIME type does not match the type
	// detected from the entry's contents.
	InputErrorMismatch InputErrorKind = "mismatch"
	// InputErrorBudget: an entry or the whole request exceeds a size or
	// count limit.
	InputErrorBudget InputErrorKind = "budget"
	// InputErrorFileState: a referenced file is not usable: processing,
	// failed, expired, deleted or unknown to the credentials.
	InputErrorFileState InputErrorKind = "file_state"
)

// File reference shape: "files/<id>" where id is 1..40 lowercase
// alphanumeric characters or dashes, not starting or ending with a dash.
// A service file URI is the same resource name prefixed by
// "https://generativelanguage.googleapis.com/".
var (
	fileNamePattern = regexp.MustCompile(`^files/[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$`)
	fileURIPattern  = regexp.MustCompile(`^https://generativelanguage\.googleapis\.com/(files/[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?)$`)
)

// BudgetItem names the limit that an [InputError] of kind [InputErrorBudget]
// tripped.
type BudgetItem string

const (
	// BudgetItemPerEntry is the size limit on a single entry.
	BudgetItemPerEntry BudgetItem = "per_entry"
	// BudgetItemPerModality is the per-modality size or count limit for
	// the whole request.
	BudgetItemPerModality BudgetItem = "per_modality"
	// BudgetItemTotal is the aggregate size or count limit for the whole
	// request across modalities.
	BudgetItemTotal BudgetItem = "total"
)

// BudgetMetric says whether a budget limit is measured in bytes or entries.
type BudgetMetric string

const (
	MetricBytes BudgetMetric = "bytes"
	MetricCount BudgetMetric = "entries"
)

// InputLocation locates an entry within a request. Indexes are zero-based.
//
// ContentIndex indexes the request's contents (including chat history, which
// is re-sent on every turn). PartIndex indexes the part inside that content.
// A location inside the system instruction or a tool declaration reports
// Scope instead and leaves the indexes set to -1.
type InputLocation struct {
	// Scope is "contents", "system_instruction" or "tools".
	Scope string
	// ContentIndex is the index of the content in Scope, or -1.
	ContentIndex int
	// PartIndex is the index of the part within its content, or -1.
	PartIndex int
}

func (l InputLocation) String() string {
	switch l.Scope {
	case scopeSystem:
		return "system_instruction"
	case scopeTools:
		return "tools"
	default:
		return fmt.Sprintf("contents[%d].parts[%d]", l.ContentIndex, l.PartIndex)
	}
}

const (
	scopeContents = "contents"
	scopeSystem   = "system_instruction"
	scopeTools    = "tools"
)

// InputError is the single error type returned for every local, pre-request
// input failure. Its fields make the failure programmatically distinguishable:
// category, modality, source (inline bytes or handle reference), entry
// location, declared and effective MIME types, size and the budget item that
// was triggered.
//
// Use errors.As to inspect it, and [InputError.Unwrap] to reach any wrapped
// underlying error (for example a file-service error).
type InputError struct {
	// Kind is the mutually exclusive error category.
	Kind InputErrorKind
	// Modality is the entry's modality, or [ModalityUnknown] if it could
	// not be determined.
	Modality InputModality
	// Source is how the entry was supplied.
	Source InputSource
	// Location locates the entry within the request.
	Location InputLocation
	// DeclaredMIME is the MIME type the caller supplied, if any.
	DeclaredMIME string
	// EffectiveMIME is the type detected from the contents, or the
	// registered type of a reference. Empty when unknown.
	EffectiveMIME string
	// Size is the entry size in bytes, or -1 when unknown (for example a
	// reference whose metadata is not held locally).
	Size int64
	// BudgetItem identifies which limit was tripped; only set for
	// [InputErrorBudget].
	BudgetItem BudgetItem
	// Metric is the unit of the tripped budget limit.
	Metric BudgetMetric
	// Limit is the configured value of the tripped limit.
	Limit int64
	// OverBy is how far the entry or aggregate exceeds the limit.
	OverBy int64
	// FileState is the lifecycle state of a referenced file; only set for
	// [InputErrorFileState].
	FileState FileState
	// Handle is the truncated, safe-to-display form of the referenced
	// file handle. Never contains raw bytes.
	Handle string

	msg string
	err error
}

func (e *InputError) Error() string {
	var b strings.Builder
	b.WriteString("genai: ")
	b.WriteString(string(e.Kind))
	b.WriteString(" input at ")
	b.WriteString(e.Location.String())
	if e.Modality != "" && e.Modality != ModalityUnknown {
		fmt.Fprintf(&b, " (modality=%s", e.Modality)
		if e.Source != "" {
			fmt.Fprintf(&b, ", source=%s", e.Source)
		}
		b.WriteString(")")
	} else if e.Source != "" {
		fmt.Fprintf(&b, " (source=%s)", e.Source)
	}
	if e.DeclaredMIME != "" || e.EffectiveMIME != "" {
		fmt.Fprintf(&b, ": declared MIME %q, effective MIME %q", e.DeclaredMIME, e.EffectiveMIME)
	}
	if e.Size >= 0 {
		fmt.Fprintf(&b, ": size %d bytes", e.Size)
	}
	if e.Kind == InputErrorBudget {
		fmt.Fprintf(&b, ": %s %s limit %d exceeded by %d", e.BudgetItem, e.Metric, e.Limit, e.OverBy)
	}
	if e.Handle != "" {
		fmt.Fprintf(&b, ": handle %s", e.Handle)
	}
	if e.FileState != FileStateUnspecified {
		fmt.Fprintf(&b, ": file state %s", e.FileState)
	}
	if e.msg != "" {
		b.WriteString(": ")
		b.WriteString(e.msg)
	}
	return b.String()
}

// Unwrap returns the wrapped underlying error, if any.
func (e *InputError) Unwrap() error { return e.err }

// ModalityStats holds the observed counts for one modality. No field ever
// contains raw bytes or file contents.
type ModalityStats struct {
	// Entries is the number of entries of this modality.
	Entries int
	// Bytes is the summed size in bytes. Entries whose size is unknown
	// contribute zero here; UnknownSizeEntries records how many did so.
	Bytes int64
	// UnknownSizeEntries is the number of entries whose size could not be
	// determined locally.
	UnknownSizeEntries int
}

// InputReport is an observation summary of one assembly pass. It contains
// per-modality counts, aggregate bytes, remaining budget and the rejected
// entries with their reasons. It never contains raw bytes or file contents,
// and file handles are truncated.
type InputReport struct {
	// PerModality holds the stats keyed by [InputModality].
	PerModality map[InputModality]ModalityStats
	// TotalEntries is the total number of media and text entries examined.
	TotalEntries int
	// TotalBytes is the summed size of entries whose size is known.
	TotalBytes int64
	// RemainingBytes is the remaining total byte budget. It is negative
	// when the budget is exceeded.
	RemainingBytes int64
	// RemainingEntries is the remaining total entry budget. It is negative
	// when the budget is exceeded.
	RemainingEntries int
	// Rejected lists one [InputError] per rejected entry. A non-empty
	// Rejected means the request must not be sent.
	Rejected []*InputError
}

// newInputError builds an InputError through one constructor so that all
// error wording and field population is shared.
func newInputError(kind InputErrorKind, loc InputLocation, msg string, wrap error, set func(*InputError)) *InputError {
	e := &InputError{
		Kind:     kind,
		Modality: ModalityUnknown,
		Source:   "",
		Location: loc,
		Size:     -1,
		msg:      msg,
		err:      wrap,
	}
	if set != nil {
		set(e)
	}
	return e
}

// truncateHandle returns a display form of a handle that cannot leak long
// user-supplied strings.
func truncateHandle(h string) string {
	const max = 24
	if len(h) <= max {
		return h
	}
	return h[:max] + "..."
}

// ---- MIME type support -----------------------------------------------------

// supportedMIMEs maps an accepted MIME type to its modality. The single map
// is the only place thresholds/types are defined, so inline and reference
// paths cannot drift.
var supportedMIMEs = map[string]InputModality{
	// Images.
	"image/png":  ModalityImage,
	"image/jpeg": ModalityImage,
	"image/webp": ModalityImage,
	"image/heic": ModalityImage,
	"image/heif": ModalityImage,
	// Audio.
	"audio/wav":       ModalityAudio,
	"audio/x-wav":     ModalityAudio,
	"audio/mpeg":      ModalityAudio,
	"audio/mp3":       ModalityAudio,
	"audio/aiff":      ModalityAudio,
	"audio/x-aiff":    ModalityAudio,
	"audio/aac":       ModalityAudio,
	"audio/ogg":       ModalityAudio,
	"audio/flac":      ModalityAudio,
	"audio/x-flac":    ModalityAudio,
	"application/ogg": ModalityAudio,
	"audio/webm":      ModalityAudio,
	// Video.
	"video/mp4":        ModalityVideo,
	"video/quicktime":  ModalityVideo,
	"video/mpeg":       ModalityVideo,
	"video/webm":       ModalityVideo,
	"video/x-matroska": ModalityVideo,
	"video/3gpp":       ModalityVideo,
	"video/3gpp2":      ModalityVideo,
	// Documents.
	"application/pdf":             ModalityDocument,
	"application/x-tika-msoffice": ModalityDocument,
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document":   ModalityDocument,
	"application/vnd.openxmlformats-officedocument.presentationml.presentation": ModalityDocument,
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":         ModalityDocument,
	"text/plain": ModalityDocument,
	"text/html":  ModalityDocument,
	"text/css":   ModalityDocument,
}

// mimeSyntaxPattern accepts well-formed type/subject tokens per RFC 6838.
var mimeSyntaxPattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]*/[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]*$`)

// validMIMESyntax reports whether m is a syntactically valid MIME type.
func validMIMESyntax(m string) bool {
	return mimeSyntaxPattern.MatchString(m)
}

// sniffMIME determines a MIME type from raw bytes using only the standard
// library plus a small magic-number table for container formats that
// [http.DetectContentType] does not recognize. It returns "" when nothing
// can be determined.
func sniffMIME(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	if m := magicMIME(data); m != "" {
		return m
	}
	// http.DetectContentType always returns a usable string (falling back
	// to application/octet-stream), so normalize an undecidable result to
	// "".
	m := http.DetectContentType(data)
	if m == "application/octet-stream" {
		return ""
	}
	// DetectContentType may append parameters (e.g. "text/plain; charset=utf-8").
	if i := strings.IndexByte(m, ';'); i >= 0 {
		m = strings.TrimSpace(m[:i])
	}
	return m
}

// magicMIME recognizes a handful of container signatures.
func magicMIME(data []byte) string {
	switch {
	case len(data) >= 4 && string(data[:4]) == "fLaC":
		return "audio/flac"
	case len(data) >= 3 && string(data[:3]) == "ID3":
		return "audio/mpeg"
	case len(data) >= 2 && data[0] == 0xFF && (data[1]&0xE0) == 0xE0:
		return "audio/mpeg"
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE":
		return "audio/wav"
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return "image/webp"
	case len(data) >= 12 && string(data[:4]) == "OggS":
		return "audio/ogg"
	case len(data) >= 8 && string(data[4:8]) == "ftyp":
		return mp4LikeMIME(data)
	case len(data) >= 4 && string(data[:4]) == "\x1A\x45\xDF\xA3": // EBML / Matroska, WebM
		return "video/x-matroska"
	case len(data) >= 4 && string(data[:4]) == "%PDF":
		return "application/pdf"
	}
	return ""
}

// mp4LikeMIME distinguishes a QuickTime file from an ISO base media (MP4)
// file using the ftyp major brand.
func mp4LikeMIME(data []byte) string {
	if len(data) < 12 {
		return "video/mp4"
	}
	brand := string(data[8:12])
	if brand == "qt  " {
		return "video/quicktime"
	}
	return "video/mp4"
}

// ---- Local budget thresholds ----------------------------------------------
//
// These are conservative client-side defaults mirroring the documented
// service limits. They are variables (not constants) so tests can exercise
// every budget branch with small fixtures; production code never mutates
// them. Inline bytes and references share the exact same values.
var (
	// Per-entry size limits in bytes, by modality.
	maxEntryBytes = map[InputModality]int64{
		ModalityImage:    10 << 20, // 10 MiB
		ModalityAudio:    20 << 20, // 20 MiB
		ModalityVideo:    20 << 20, // 20 MiB
		ModalityDocument: 10 << 20, // 10 MiB
	}
	// Per-modality entry counts for one request.
	maxModalityEntries = map[InputModality]int64{
		ModalityImage:    16,
		ModalityAudio:    16,
		ModalityVideo:    16,
		ModalityDocument: 16,
	}
	// Aggregate limits for one request across all modalities.
	maxTotalMediaBytes   int64 = 60 << 20 // 60 MiB
	maxTotalMediaEntries int64 = 64
)

// ---- File registry --------------------------------------------------------

// fileRecord is the locally known metadata for an uploaded file. The
// registry is populated by UploadFile and GetFile, allowing assembly-time
// lifecycle checks without a network round trip.
type fileRecord struct {
	name           string
	uri            string
	mime           string
	size           int64
	state          FileState
	expirationTime int64 // unix seconds; zero if unknown
}

// fileRegistry is an unexported per-Client cache of file metadata.
type fileRegistry struct {
	mu     sync.RWMutex
	byName map[string]*fileRecord
	byURI  map[string]*fileRecord
}

func newFileRegistry() *fileRegistry {
	return &fileRegistry{
		byName: map[string]*fileRecord{},
		byURI:  map[string]*fileRecord{},
	}
}

func (r *fileRegistry) put(rec *fileRecord) {
	if rec == nil || rec.name == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byName[rec.name] = rec
	if rec.uri != "" {
		r.byURI[rec.uri] = rec
	}
}

func (r *fileRegistry) delete(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rec, ok := r.byName[name]; ok {
		delete(r.byURI, rec.uri)
	}
	delete(r.byName, name)
}

// lookupByNameOrURI finds a record by either handle shape.
func (r *fileRegistry) lookup(handle string) *fileRecord {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if rec, ok := r.byName[handle]; ok {
		return rec
	}
	return r.byURI[handle]
}

// rememberFile stores the metadata of a [File] returned by the service.
func (c *Client) rememberFile(f *File) {
	if c == nil || c.files == nil || f == nil || f.Name == "" {
		return
	}
	c.files.put(&fileRecord{
		name:           f.Name,
		uri:            f.URI,
		mime:           f.MIMEType,
		size:           f.SizeBytes,
		state:          f.State,
		expirationTime: f.ExpirationTime.Unix(),
	})
}

// ---- Assembly -------------------------------------------------------------

// assembledEntry is one validated media/text entry, in original order.
type assembledEntry struct {
	loc       InputLocation
	modality  InputModality
	source    InputSource
	mime      string
	size      int64
	sizeKnown bool
	handle    string
}

// inputValidator carries the state of one assembly pass.
type inputValidator struct {
	client   *Client
	entries  []assembledEntry
	rejected []*InputError
	// seenReferences de-duplicates repeated references to the same handle:
	// the handle is counted once for content bytes, but each use still
	// counts as an entry.
	seenReferences map[string]bool
}

// validateMultimodalInput runs the shared pipeline over every multimodal
// surface of a request: the contents, the system instruction and the tool
// declarations. It returns an observation report and an error which is the
// first [InputError] when one or more entries are rejected.
//
// On success, inline [Blob] parts with an empty MIMEType have been filled in
// with the type sniffed from their bytes, so serialization stays identical to
// a caller-supplied type.
func (c *Client) validateMultimodalInput(system *Content, contents []*Content, tools []*Tool) (*InputReport, error) {
	v := &inputValidator{
		client:         c,
		seenReferences: map[string]bool{},
	}
	if system != nil {
		v.walkContent(system, InputLocation{Scope: scopeSystem, ContentIndex: -1, PartIndex: -1})
	}
	for i, content := range contents {
		if content == nil {
			continue
		}
		v.walkContent(content, InputLocation{Scope: scopeContents, ContentIndex: i, PartIndex: -1})
	}
	for i, tool := range tools {
		v.walkTool(tool, InputLocation{Scope: scopeTools, ContentIndex: i, PartIndex: -1})
	}
	report := v.meter()
	if len(v.rejected) > 0 {
		return report, v.rejected[0]
	}
	return report, nil
}

// walkContent validates each part of one Content in order. When an inline
// [Blob] had no declared MIME type, the type inferred from its bytes is
// written back into the content so the serialized request carries it.
func (v *inputValidator) walkContent(c *Content, loc InputLocation) {
	for i, p := range c.Parts {
		partLoc := loc
		partLoc.PartIndex = i
		inferred := v.validatePart(p, partLoc)
		if inferred != "" {
			if b, ok := p.(Blob); ok && b.MIMEType == "" {
				b.MIMEType = inferred
				c.Parts[i] = b
			}
		}
	}
}

// walkTool covers multimodal content inside tool declarations. The current
// tool surface carries only function declarations and code execution, so it
// has no media parts, but the traversal keeps the guarantee that tools go
// through the same pipeline if media is added later.
func (v *inputValidator) walkTool(t *Tool, loc InputLocation) {
	if t == nil {
		return
	}
	// FunctionDeclarations and CodeExecution hold no Parts today. This
	// switch exists so new media-bearing tool fields cannot bypass the
	// pipeline silently.
	switch {
	case t.CodeExecution != nil:
		// No media; nothing to validate.
	case len(t.FunctionDeclarations) > 0:
		// Schemas and descriptions are text-only; nothing to validate.
	}
}

// validatePart dispatches a single Part to the inline or reference path.
// It returns the MIME type inferred for an inline [Blob] that had none,
// or "" otherwise.
func (v *inputValidator) validatePart(p Part, loc InputLocation) string {
	switch part := p.(type) {
	case nil:
		v.reject(newInputError(InputErrorMalformed, loc, "nil part", nil, nil))
	case Text:
		v.entries = append(v.entries, assembledEntry{
			loc:       loc,
			modality:  ModalityText,
			source:    SourceInline,
			size:      int64(len(string(part))),
			sizeKnown: true,
		})
	case Blob:
		return v.validateInline(part, loc)
	case *Blob:
		if part == nil {
			v.reject(newInputError(InputErrorMalformed, loc, "nil blob", nil, nil))
			return ""
		}
		return v.validateInline(*part, loc)
	case FileData:
		v.validateReference(part, loc)
	case *FileData:
		if part == nil {
			v.reject(newInputError(InputErrorMalformed, loc, "nil file data", nil, nil))
			return ""
		}
		v.validateReference(*part, loc)
	default:
		// FunctionCall, FunctionResponse, ExecutableCode,
		// CodeExecutionResult and any future non-media part carry no
		// multimodal payload and pass through untouched.
	}
	return ""
}

// validateInline handles an inline [Blob]: sniff when undeclared, reject on
// declaration mismatch or unsupported type, then meter the bytes. It returns
// the inferred MIME type when the blob had none.
func (v *inputValidator) validateInline(b Blob, loc InputLocation) string {
	declared := strings.TrimSpace(b.MIMEType)
	if len(b.Data) == 0 {
		v.reject(newInputError(InputErrorMalformed, loc, "inline blob has no data", nil, func(e *InputError) {
			e.Source = SourceInline
			e.DeclaredMIME = declared
		}))
		return ""
	}
	sniffed := sniffMIME(b.Data)

	effective := declared
	if effective == "" {
		// Undeclared: infer a valid type from the contents.
		if sniffed == "" {
			v.reject(newInputError(InputErrorUnsupported, loc,
				"cannot determine MIME type from inline bytes; set Blob.MIMEType explicitly", nil,
				func(e *InputError) {
					e.Source = SourceInline
					e.Size = int64(len(b.Data))
				}))
			return ""
		}
		effective = sniffed
	} else {
		// Declared: the syntax, support and the declaration/content
		// agreement are checked in a fixed, mutually exclusive order.
		if !validMIMESyntax(declared) {
			v.reject(newInputError(InputErrorMalformed, loc,
				"declared MIME type is not a valid MIME type", nil,
				func(e *InputError) {
					e.Source = SourceInline
					e.DeclaredMIME = declared
					e.EffectiveMIME = sniffed
					e.Size = int64(len(b.Data))
				}))
			return ""
		}
		if _, ok := supportedMIMEs[declared]; !ok {
			v.reject(newInputError(InputErrorUnsupported, loc,
				"declared MIME type is not a supported input type", nil,
				func(e *InputError) {
					e.Source = SourceInline
					e.DeclaredMIME = declared
					e.EffectiveMIME = sniffed
					e.Size = int64(len(b.Data))
				}))
			return ""
		}
		if sniffed != "" && !sameMIMEFamily(declared, sniffed) {
			v.reject(newInputError(InputErrorMismatch, loc,
				"declared MIME type does not match the type detected from the bytes", nil,
				func(e *InputError) {
					e.Source = SourceInline
					e.DeclaredMIME = declared
					e.EffectiveMIME = sniffed
					e.Modality = supportedMIMEs[declared]
					e.Size = int64(len(b.Data))
				}))
			return ""
		}
	}

	modality, ok := supportedMIMEs[effective]
	if !ok {
		// The sniffed type is recognizable but not accepted (executables,
		// archives, ...). Distinct from malformed and mismatch.
		v.reject(newInputError(InputErrorUnsupported, loc,
			"detected MIME type is not a supported input type", nil,
			func(e *InputError) {
				e.Source = SourceInline
				e.DeclaredMIME = declared
				e.EffectiveMIME = effective
				e.Size = int64(len(b.Data))
			}))
		return ""
	}

	// The inferred type is written back onto the caller's part by
	// walkContent; do not mutate the local copy here.

	v.entries = append(v.entries, assembledEntry{
		loc:       loc,
		modality:  modality,
		source:    SourceInline,
		mime:      effective,
		size:      int64(len(b.Data)),
		sizeKnown: true,
	})
	// Report an inferred type only when the caller did not declare one.
	if declared == "" {
		return effective
	}
	return ""
}

// validateReference handles a [FileData] handle: shape, registry lifecycle
// state and type support, then meters it with the same thresholds as inline
// bytes.
func (v *inputValidator) validateReference(fd FileData, loc InputLocation) {
	declared := strings.TrimSpace(fd.MIMEType)
	handle, ok := canonicalHandle(fd.URI)
	if !ok {
		v.reject(newInputError(InputErrorMalformed, loc,
			"file reference must be a handle (files/<id>) or URI returned by this service's UploadFile; local paths, external links, empty and malformed handles are not accepted",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.DeclaredMIME = declared
				e.Handle = truncateHandle(fd.URI)
			}))
		return
	}

	var rec *fileRecord
	if v.client != nil && v.client.files != nil {
		rec = v.client.files.lookup(handle)
	}
	if rec == nil {
		// Unknown to the local registry. We cannot distinguish
		// nonexistent, deleted and foreign resources without a network
		// call, and must not allow handle probing: report a single
		// outward category.
		v.reject(newInputError(InputErrorFileState, loc,
			"file reference is not available to these credentials; use Client.UploadFile and wait until the file is active before referencing it",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.DeclaredMIME = declared
				e.Handle = truncateHandle(handle)
				e.FileState = FileStateUnspecified
			}))
		return
	}

	// Lifecycle checks happen at assembly time, not on the remote.
	switch rec.state {
	case FileStateProcessing:
		v.reject(newInputError(InputErrorFileState, loc,
			"file is still processing; query it with Client.GetFile or wait until its state becomes FileStateActive, then retry",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.Handle = truncateHandle(handle)
				e.FileState = FileStateProcessing
				e.Size = rec.size
			}))
		return
	case FileStateFailed:
		v.reject(newInputError(InputErrorFileState, loc,
			"file failed processing and cannot be recovered; upload it again",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.Handle = truncateHandle(handle)
				e.FileState = FileStateFailed
				e.Size = rec.size
			}))
		return
	case FileStateExpired:
		v.reject(newInputError(InputErrorFileState, loc,
			"file has expired and cannot be recovered; upload it again",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.Handle = truncateHandle(handle)
				e.FileState = FileStateExpired
				e.Size = rec.size
			}))
		return
	case FileStateDeleted:
		// Deleted, nonexistent and foreign-owned handles share one
		// category so the error cannot be used to probe other resources.
		v.reject(newInputError(InputErrorFileState, loc,
			"file reference is not available to these credentials",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.Handle = truncateHandle(handle)
				e.FileState = FileStateDeleted
				e.Size = rec.size
			}))
		return
	case FileStateActive:
		// Continue below.
	default:
		v.reject(newInputError(InputErrorFileState, loc,
			"file is not active",
			nil, func(e *InputError) {
				e.Source = SourceReference
				e.Handle = truncateHandle(handle)
				e.FileState = rec.state
				e.Size = rec.size
			}))
		return
	}

	effective := declared
	if effective == "" {
		effective = rec.mime
	}
	if effective != "" {
		if !validMIMESyntax(effective) {
			v.reject(newInputError(InputErrorMalformed, loc,
				"file MIME type is not a valid MIME type", nil,
				func(e *InputError) {
					e.Source = SourceReference
					e.DeclaredMIME = declared
					e.EffectiveMIME = rec.mime
					e.Handle = truncateHandle(handle)
				}))
			return
		}
		if _, ok := supportedMIMEs[effective]; !ok {
			v.reject(newInputError(InputErrorUnsupported, loc,
				"file MIME type is not a supported input type", nil,
				func(e *InputError) {
					e.Source = SourceReference
					e.DeclaredMIME = declared
					e.EffectiveMIME = effective
					e.Handle = truncateHandle(handle)
					e.Size = rec.size
				}))
			return
		}
	}
	modality := supportedMIMEs[effective]
	if modality == "" {
		modality = ModalityUnknown
	}

	entry := assembledEntry{
		loc:      loc,
		modality: modality,
		source:   SourceReference,
		mime:     effective,
		handle:   handle,
	}
	if rec.size > 0 {
		entry.size = rec.size
		entry.sizeKnown = true
	} else {
		entry.size = 0
		entry.sizeKnown = false
	}
	v.entries = append(v.entries, entry)
}

// canonicalHandle accepts either a "files/<id>" name or a service URI and
// returns the canonical "files/<id>" handle.
func canonicalHandle(h string) (string, bool) {
	h = strings.TrimSpace(h)
	if h == "" {
		return "", false
	}
	if fileNamePattern.MatchString(h) {
		return h, true
	}
	if m := fileURIPattern.FindStringSubmatch(h); m != nil {
		return m[1], true
	}
	return "", false
}

// sameMIMEFamily reports whether two MIME types should be treated as
// equivalent for declaration-mismatch checks. It normalizes a handful of
// legacy aliases (audio/mpeg vs audio/mp3, audio/wav vs audio/x-wav, ...).
func sameMIMEFamily(a, b string) bool {
	if a == b {
		return true
	}
	aliases := map[string]string{
		"audio/x-wav":  "audio/wav",
		"audio/mp3":    "audio/mpeg",
		"audio/x-flac": "audio/flac",
		"audio/x-aiff": "audio/aiff",
	}
	return aliases[a] == b || aliases[b] == a
}

// reject records a rejection and guarantees the entry is never metered.
func (v *inputValidator) reject(e *InputError) {
	v.rejected = append(v.rejected, e)
}

// meter applies the per-entry, per-modality and total budgets to all
// accepted entries and builds the observation report. Rejected entries are
// included in the report (with their reasons) but never count towards
// budget consumption.
func (v *inputValidator) meter() *InputReport {
	report := &InputReport{
		PerModality: map[InputModality]ModalityStats{},
		Rejected:    v.rejected,
	}

	// Rejected entries still appear in the report for observability, but
	// with no stats.
	report.TotalEntries = len(v.entries)

	// Track unique references: a repeated handle counts its content bytes
	// only once (it is one cached object), though each use is an entry.
	countedBytes := map[string]bool{}
	modalityCount := map[InputModality]int64{}
	modalityBytes := map[InputModality]int64{}
	modalityUnknown := map[InputModality]int{}

	// Per-entry limits first, so an oversized single part is reported as
	// per_entry rather than blamed on the aggregate.
	for _, en := range v.entries {
		if en.modality == ModalityText || en.modality == ModalityUnknown {
			// Text has no media byte cap locally; unknown modality is
			// reported separately via the entry's own rejection path.
		} else if en.sizeKnown {
			if limit, ok := maxEntryBytes[en.modality]; ok && en.size > limit {
				v.reject(newInputError(InputErrorBudget, en.loc,
					"entry exceeds the per-entry size limit; for inline data, upload the file with Client.UploadFile and pass a FileData reference instead",
					nil, func(e *InputError) {
						e.Modality = en.modality
						e.Source = en.source
						e.EffectiveMIME = en.mime
						e.Size = en.size
						e.BudgetItem = BudgetItemPerEntry
						e.Metric = MetricBytes
						e.Limit = limit
						e.OverBy = en.size - limit
						e.Handle = truncateHandle(en.handle)
					}))
				continue
			}
		}
		modalityCount[en.modality]++
		if en.sizeKnown {
			// A repeated reference to the same handle counts its content
			// bytes only once: it is one cached object. Every use still
			// counts as an entry (the increment above), and every inline
			// entry is distinct and always counted.
			if en.source == SourceReference {
				key := "ref:" + en.handle
				if countedBytes[key] {
					continue
				}
				countedBytes[key] = true
			}
			modalityBytes[en.modality] += en.size
			report.TotalBytes += en.size
		} else if en.source == SourceReference {
			modalityUnknown[en.modality]++
		}
	}

	// Per-modality limits (count and bytes).
	for modality, count := range modalityCount {
		if limit, ok := maxModalityEntries[modality]; ok && count > limit {
			// Attribute to the first entry of that modality for a stable
			// location.
			loc := v.firstLocation(modality)
			v.reject(newInputError(InputErrorBudget, loc,
				"request exceeds the per-modality entry count limit", nil,
				func(e *InputError) {
					e.Modality = modality
					e.BudgetItem = BudgetItemPerModality
					e.Metric = MetricCount
					e.Limit = limit
					e.OverBy = count - limit
				}))
		}
	}
	for modality, bytes := range modalityBytes {
		entryLimit, ok := maxEntryBytes[modality]
		if !ok {
			continue
		}
		// The per-modality byte budget equals the per-entry byte cap
		// scaled by the modality entry allowance; derive it once.
		modLimit := entryLimit * maxModalityEntries[modality]
		if bytes > modLimit {
			loc := v.firstLocation(modality)
			v.reject(newInputError(InputErrorBudget, loc,
				"request exceeds the per-modality total size limit", nil,
				func(e *InputError) {
					e.Modality = modality
					e.BudgetItem = BudgetItemPerModality
					e.Metric = MetricBytes
					e.Limit = modLimit
					e.OverBy = bytes - modLimit
				}))
		}
	}

	// Total limits across modalities.
	var totalMediaCount int64
	for modality, count := range modalityCount {
		if modality != ModalityText {
			totalMediaCount += count
		}
	}
	if totalMediaCount > maxTotalMediaEntries {
		loc := v.firstMediaLocation()
		v.reject(newInputError(InputErrorBudget, loc,
			"request exceeds the total media entry count limit", nil,
			func(e *InputError) {
				e.BudgetItem = BudgetItemTotal
				e.Metric = MetricCount
				e.Limit = maxTotalMediaEntries
				e.OverBy = totalMediaCount - maxTotalMediaEntries
			}))
	}
	if report.TotalBytes > maxTotalMediaBytes {
		loc := v.firstMediaLocation()
		v.reject(newInputError(InputErrorBudget, loc,
			"request exceeds the total media size limit; upload large content with Client.UploadFile and reference it",
			nil, func(e *InputError) {
				e.BudgetItem = BudgetItemTotal
				e.Metric = MetricBytes
				e.Limit = maxTotalMediaBytes
				e.OverBy = report.TotalBytes - maxTotalMediaBytes
			}))
	}

	// Fill the report stats from the already de-duplicated aggregates.
	// Rejected entries never entered the aggregates because per-entry
	// rejection continues the loop above, so every count here is accepted.
	for modality, count := range modalityCount {
		st := report.PerModality[modality]
		st.Entries = int(count)
		st.Bytes = modalityBytes[modality]
		st.UnknownSizeEntries = modalityUnknown[modality]
		report.PerModality[modality] = st
	}
	report.RemainingBytes = maxTotalMediaBytes - report.TotalBytes
	report.RemainingEntries = int(maxTotalMediaEntries - totalMediaCount)
	report.Rejected = v.rejected
	return report
}

func (v *inputValidator) firstLocation(modality InputModality) InputLocation {
	for _, en := range v.entries {
		if en.modality == modality {
			return en.loc
		}
	}
	return InputLocation{Scope: scopeContents, ContentIndex: -1, PartIndex: -1}
}

func (v *inputValidator) firstMediaLocation() InputLocation {
	for _, en := range v.entries {
		if en.modality != ModalityText {
			return en.loc
		}
	}
	return InputLocation{Scope: scopeContents, ContentIndex: -1, PartIndex: -1}
}

// AsInputError extracts an [InputError] from err, or returns nil.
func AsInputError(err error) *InputError {
	var ie *InputError
	if errors.As(err, &ie) {
		return ie
	}
	return nil
}

// LastInputReport returns the [InputReport] produced by the most recent
// assembly pass performed through this model, or nil if none has run. It
// never contains raw bytes. This is a convenience for callers that want the
// observation summary alongside the returned error.
func (m *GenerativeModel) LastInputReport() *InputReport {
	if m == nil {
		return nil
	}
	return m.lastReport
}

// validateModelInput is the single hook every request-building path calls
// before serializing. It keeps a copy of the report on the model for
// observability. A zero-value model (no client) skips validation; its
// serialization still runs and retains the pre-existing panic-to-error
// behavior.
func (m *GenerativeModel) validateModelInput(contents []*Content) error {
	if m == nil || m.c == nil {
		return nil
	}
	report, err := m.c.validateMultimodalInput(m.SystemInstruction, contents, m.Tools)
	m.lastReport = report
	return err
}
