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

package genai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	pb "cloud.google.com/go/ai/generativelanguage/apiv1beta/generativelanguagepb"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/proto"
)

// ---- Test fixtures --------------------------------------------------------

// Small, valid magic-byte payloads.
var (
	pngBytes  = append([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, make([]byte, 32)...)
	jpegBytes = append([]byte{0xFF, 0xD8, 0xFF, 0xE0}, make([]byte, 32)...)
	pdfBytes  = append([]byte("%PDF-1.7\n"), make([]byte, 32)...)
	mp3Bytes  = append([]byte("ID3"), make([]byte, 64)...)
	mp4Bytes  = append(append([]byte{0, 0, 0, 0x18}, []byte("ftyp")...), []byte("mp42")...)
	// An executable/archive-like payload that is not a supported type.
	exeBytes = append([]byte{'M', 'Z'}, make([]byte, 64)...)
)

// newTestClient builds a Client without a network connection. The file
// registry is the only piece the assembly pipeline uses, so no transport is
// exercised.
func newTestClient(t *testing.T) *Client {
	t.Helper()
	c, err := NewClient(context.Background(), option.WithAPIKey("test-key"),
		option.WithEndpoint("https://example.invalid"))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// registerFile injects a file record directly, mirroring what UploadFile /
// GetFile would store.
func registerFile(c *Client, name, uri, mime string, size int64, state FileState) {
	c.files.put(&fileRecord{name: name, uri: uri, mime: mime, size: size, state: state})
}

const (
	testHandle = "files/abc-123"
	testURI    = "https://generativelanguage.googleapis.com/files/abc-123"
)

// withThresholds temporarily swaps the budget variables for a test and
// restores them afterwards.
func withThresholds(t *testing.T, perEntry, perModCount, totalBytes, totalCount int64) {
	t.Helper()
	oldEntry := maxEntryBytes
	oldModCount := maxModalityEntries
	oldTotalBytes := maxTotalMediaBytes
	oldTotalCount := maxTotalMediaEntries
	maxEntryBytes = map[InputModality]int64{
		ModalityImage:    perEntry,
		ModalityAudio:    perEntry,
		ModalityVideo:    perEntry,
		ModalityDocument: perEntry,
	}
	maxModalityEntries = map[InputModality]int64{
		ModalityImage:    perModCount,
		ModalityAudio:    perModCount,
		ModalityVideo:    perModCount,
		ModalityDocument: perModCount,
	}
	maxTotalMediaBytes = totalBytes
	maxTotalMediaEntries = totalCount
	t.Cleanup(func() {
		maxEntryBytes = oldEntry
		maxModalityEntries = oldModCount
		maxTotalMediaBytes = oldTotalBytes
		maxTotalMediaEntries = oldTotalCount
	})
}

// ---- 1. Type detection ----------------------------------------------------

func TestSniffMIME(t *testing.T) {
	for _, test := range []struct {
		name string
		data []byte
		want string
	}{
		{"png", pngBytes, "image/png"},
		{"jpeg", jpegBytes, "image/jpeg"},
		{"pdf", pdfBytes, "application/pdf"},
		{"mp3", mp3Bytes, "audio/mpeg"},
		{"mp4", mp4Bytes, "video/mp4"},
		{"empty", nil, ""},
		{"undecidable", []byte{0x00, 0x01, 0x02, 0x03}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := sniffMIME(test.data); got != test.want {
				t.Errorf("sniffMIME = %q, want %q", got, test.want)
			}
		})
	}
}

func TestCanonicalHandle(t *testing.T) {
	for _, test := range []struct {
		in     string
		want   string
		wantOK bool
	}{
		{testHandle, testHandle, true},
		{testURI, testHandle, true},
		{"", "", false},
		{"/local/path/img.png", "", false},
		{"https://evil.example.com/files/abc-123", "", false},
		{"files/", "", false},
		{"files/UPPER", "", false},
		{"files/-bad-", "", false},
		{"random-string", "", false},
		{"http://generativelanguage.googleapis.com/files/abc-123", "", false},
	} {
		t.Run(test.in, func(t *testing.T) {
			got, ok := canonicalHandle(test.in)
			if ok != test.wantOK || got != test.want {
				t.Errorf("canonicalHandle(%q) = %q,%v; want %q,%v", test.in, got, ok, test.want, test.wantOK)
			}
		})
	}
}

// ---- 2. Inline vs reference matrix ---------------------------------------

func TestInlineReferenceMatrix(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "image/png", int64(len(pngBytes)), FileStateActive)
	withThresholds(t, 1<<30, 16, 1<<30, 64)

	// The same content supplied inline and by reference must produce the
	// same modality and budget conclusion.
	t.Run("valid-inline", func(t *testing.T) {
		report, err := c.validateMultimodalInput(nil,
			[]*Content{NewUserContent(Blob{Data: pngBytes})}, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertModalityStats(t, report, ModalityImage, 1, int64(len(pngBytes)))
	})

	t.Run("valid-reference", func(t *testing.T) {
		report, err := c.validateMultimodalInput(nil,
			[]*Content{NewUserContent(FileData{URI: testURI})}, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertModalityStats(t, report, ModalityImage, 1, int64(len(pngBytes)))
	})

	t.Run("same-conclusion", func(t *testing.T) {
		r1, e1 := c.validateMultimodalInput(nil,
			[]*Content{NewUserContent(Blob{Data: pngBytes})}, nil)
		r2, e2 := c.validateMultimodalInput(nil,
			[]*Content{NewUserContent(FileData{URI: testURI})}, nil)
		if (e1 == nil) != (e2 == nil) {
			t.Fatalf("acceptance differs: inline err=%v reference err=%v", e1, e2)
		}
		s1 := r1.PerModality[ModalityImage]
		s2 := r2.PerModality[ModalityImage]
		if s1.Entries != s2.Entries || s1.Bytes != s2.Bytes {
			t.Errorf("metering differs: inline=%+v reference=%+v", s1, s2)
		}
	})
}

// ---- 3. The three mutually exclusive type errors -------------------------

func TestTypeErrorCategories(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)

	for _, test := range []struct {
		name     string
		part     Part
		wantKind InputErrorKind
		wantLoc  InputLocation
	}{
		{
			name:     "malformed-empty-blob",
			part:     Blob{Data: nil, MIMEType: "image/png"},
			wantKind: InputErrorMalformed,
		},
		{
			name:     "malformed-bad-mime-syntax",
			part:     Blob{Data: pngBytes, MIMEType: "not a mime"},
			wantKind: InputErrorMalformed,
		},
		{
			name:     "unsupported-executable",
			part:     Blob{Data: exeBytes, MIMEType: "application/x-msdownload"},
			wantKind: InputErrorUnsupported,
		},
		{
			name:     "unsupported-archive-sniffed",
			part:     Blob{Data: exeBytes},
			wantKind: InputErrorUnsupported,
		},
		{
			name:     "mismatch-declared-jpeg-but-png",
			part:     Blob{Data: pngBytes, MIMEType: "image/jpeg"},
			wantKind: InputErrorMismatch,
		},
		{
			name:     "malformed-local-path-reference",
			part:     FileData{URI: "/etc/passwd"},
			wantKind: InputErrorMalformed,
		},
		{
			name:     "malformed-external-link-reference",
			part:     FileData{URI: "https://storage.example.com/bucket/obj"},
			wantKind: InputErrorMalformed,
		},
		{
			name:     "malformed-empty-reference",
			part:     FileData{URI: ""},
			wantKind: InputErrorMalformed,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := c.validateMultimodalInput(nil,
				[]*Content{NewUserContent(test.part)}, nil)
			ie := requireInputError(t, err)
			if ie.Kind != test.wantKind {
				t.Fatalf("kind = %s, want %s (err=%v)", ie.Kind, test.wantKind, err)
			}
			if ie.Location.ContentIndex != 0 || ie.Location.PartIndex != 0 {
				t.Errorf("location = %+v, want contents[0].parts[0]", ie.Location)
			}
		})
	}
}

// ---- 4. Mismatch carries both declared and effective type ---------------

func TestMismatchFields(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(Blob{Data: pngBytes, MIMEType: "image/jpeg"})}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorMismatch {
		t.Fatalf("kind = %s, want mismatch", ie.Kind)
	}
	if ie.DeclaredMIME != "image/jpeg" || ie.EffectiveMIME != "image/png" {
		t.Errorf("declared=%q effective=%q", ie.DeclaredMIME, ie.EffectiveMIME)
	}
	if ie.Modality != ModalityImage {
		t.Errorf("modality = %s", ie.Modality)
	}
	if ie.Size != int64(len(pngBytes)) {
		t.Errorf("size = %d", ie.Size)
	}
}

// ---- 5. Inline type inference writes back MIME ---------------------------

func TestInlineInferenceWritesBack(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	content := NewUserContent(Blob{Data: pngBytes})
	if _, err := c.validateMultimodalInput(nil, []*Content{content}, nil); err != nil {
		t.Fatal(err)
	}
	got := content.Parts[0].(Blob).MIMEType
	if got != "image/png" {
		t.Errorf("inferred MIME = %q, want image/png", got)
	}
}

// ---- 6. Budget limits -----------------------------------------------------

func TestBudgetPerEntry(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 10, 16, 1<<30, 64) // 10-byte per-entry limit
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(Blob{Data: pngBytes, MIMEType: "image/png"})}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorBudget || ie.BudgetItem != BudgetItemPerEntry || ie.Metric != MetricBytes {
		t.Fatalf("got kind=%s item=%s metric=%s", ie.Kind, ie.BudgetItem, ie.Metric)
	}
	if ie.Limit != 10 || ie.OverBy != int64(len(pngBytes))-10 {
		t.Errorf("limit=%d overby=%d", ie.Limit, ie.OverBy)
	}
	if !strings.Contains(ie.Error(), "per_entry") {
		t.Errorf("error message missing budget item: %v", ie)
	}
}

func TestBudgetTotal(t *testing.T) {
	c := newTestClient(t)
	// Generous per-entry and per-modality limits, tight total.
	withThresholds(t, 1<<30, 16, 40, 64)
	parts := []Part{
		Blob{Data: pngBytes, MIMEType: "image/png"},
		Blob{Data: jpegBytes, MIMEType: "image/jpeg"},
	}
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(parts...)}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorBudget || ie.BudgetItem != BudgetItemTotal {
		t.Fatalf("got kind=%s item=%s", ie.Kind, ie.BudgetItem)
	}
	wantOver := int64(len(pngBytes)+len(jpegBytes)) - 40
	if ie.OverBy != wantOver {
		t.Errorf("overby = %d, want %d", ie.OverBy, wantOver)
	}
}

func TestBudgetPerModalityCount(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 2, 1<<30, 64)
	parts := []Part{
		Blob{Data: pngBytes, MIMEType: "image/png"},
		Blob{Data: pngBytes, MIMEType: "image/png"},
		Blob{Data: pngBytes, MIMEType: "image/png"},
	}
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(parts...)}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorBudget || ie.BudgetItem != BudgetItemPerModality || ie.Metric != MetricCount {
		t.Fatalf("got kind=%s item=%s metric=%s", ie.Kind, ie.BudgetItem, ie.Metric)
	}
}

func TestReferenceSharesBudgetThresholds(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "image/png", 100, FileStateActive)
	withThresholds(t, 50, 16, 1<<30, 64) // reference size 100 > per-entry 50
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: testHandle})}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorBudget || ie.BudgetItem != BudgetItemPerEntry {
		t.Fatalf("got kind=%s item=%s", ie.Kind, ie.BudgetItem)
	}
	if ie.Source != SourceReference {
		t.Errorf("source = %s", ie.Source)
	}
}

// ---- 7. File lifecycle ----------------------------------------------------

func TestFileLifecycle(t *testing.T) {
	for _, test := range []struct {
		name      string
		state     FileState
		wantKind  InputErrorKind
		wantState FileState
		retryable bool
	}{
		{"processing", FileStateProcessing, InputErrorFileState, FileStateProcessing, true},
		{"failed", FileStateFailed, InputErrorFileState, FileStateFailed, false},
		{"expired", FileStateExpired, InputErrorFileState, FileStateExpired, false},
		{"deleted", FileStateDeleted, InputErrorFileState, FileStateDeleted, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := newTestClient(t)
			registerFile(c, testHandle, testURI, "image/png", 10, test.state)
			_, err := c.validateMultimodalInput(nil,
				[]*Content{NewUserContent(FileData{URI: testHandle})}, nil)
			ie := requireInputError(t, err)
			if ie.Kind != test.wantKind || ie.FileState != test.wantState {
				t.Fatalf("got kind=%s state=%s", ie.Kind, ie.FileState)
			}
			if test.retryable && !strings.Contains(ie.Error(), "retry") {
				t.Errorf("processing state should mention retry: %v", ie)
			}
			if !test.retryable && strings.Contains(ie.Error(), "retry") {
				t.Errorf("non-retryable state unexpectedly mentions retry: %v", ie)
			}
		})
	}
}

func TestUnknownReferenceIsFileStateNotProbe(t *testing.T) {
	c := newTestClient(t)
	// No registry entry: nonexistent / foreign / deleted all collapse to
	// the same category and wording.
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: "files/unknown-id-1"})}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorFileState {
		t.Fatalf("kind = %s, want file_state", ie.Kind)
	}
	if strings.Contains(ie.Error(), "not found") || strings.Contains(ie.Error(), "permission") {
		t.Errorf("error must not distinguish absence vs ownership: %v", ie)
	}
}

// ---- 8. Multi-turn history counts every turn -----------------------------

func TestChatHistoryBudget(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 2) // only 2 media entries total
	history := []*Content{
		NewUserContent(Text("first"), Blob{Data: pngBytes, MIMEType: "image/png"}),
		NewUserContent(Text("second"), Blob{Data: jpegBytes, MIMEType: "image/jpeg"}),
	}
	// The third turn re-sends both historical media entries plus one more.
	history = append(history, NewUserContent(Text("third"), Blob{Data: pngBytes, MIMEType: "image/png"}))
	_, err := c.validateMultimodalInput(nil, history, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorBudget {
		t.Fatalf("kind = %s", ie.Kind)
	}
	// The offending media is located inside the historical contents.
	if ie.Location.ContentIndex < 0 {
		t.Errorf("budget error not located to a content entry: %+v", ie.Location)
	}
}

// ---- 9. Repeated reference counts bytes once, entries each ---------------

func TestRepeatedReferenceDedup(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "image/png", 100, FileStateActive)
	withThresholds(t, 1<<30, 16, 150, 64) // total bytes 150; dedup means 100, not 300
	parts := []Part{
		FileData{URI: testHandle},
		FileData{URI: testHandle},
		FileData{URI: testHandle},
	}
	report, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(parts...)}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	st := report.PerModality[ModalityImage]
	if st.Bytes != 100 {
		t.Errorf("deduped bytes = %d, want 100", st.Bytes)
	}
	if st.Entries != 3 {
		t.Errorf("entries = %d, want 3", st.Entries)
	}
}

// ---- 10. Order preservation and mixed modalities -------------------------

func TestOrderAndMixedModalities(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "video/mp4", int64(len(mp4Bytes)), FileStateActive)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	content := NewUserContent(
		Text("look"),
		Blob{Data: pngBytes, MIMEType: "image/png"},
		Blob{Data: mp3Bytes, MIMEType: "audio/mpeg"},
		FileData{URI: testURI},
		Text("done"),
	)
	report, err := c.validateMultimodalInput(nil, []*Content{content}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Serialization order is untouched: validate via proto conversion.
	proto := content.toProto()
	if len(proto.Parts) != 5 {
		t.Fatalf("part count = %d, want 5", len(proto.Parts))
	}
	if _, ok := proto.Parts[0].Data.(*pb.Part_Text); !ok {
		t.Errorf("part 0 not text: %T", proto.Parts[0].Data)
	}
	if _, ok := proto.Parts[3].Data.(*pb.Part_FileData); !ok {
		t.Errorf("part 3 not file data: %T", proto.Parts[3].Data)
	}
	if report.PerModality[ModalityImage].Entries != 1 ||
		report.PerModality[ModalityAudio].Entries != 1 ||
		report.PerModality[ModalityVideo].Entries != 1 ||
		report.PerModality[ModalityText].Entries != 2 {
		t.Errorf("unexpected modality stats: %+v", report.PerModality)
	}
}

// ---- 11. System instruction and tools go through the pipeline ------------

func TestSystemInstructionValidated(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	sys := &Content{Parts: []Part{Blob{Data: exeBytes, MIMEType: "application/x-msdownload"}}}
	_, err := c.validateMultimodalInput(sys,
		[]*Content{NewUserContent(Text("hi"))}, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorUnsupported || ie.Location.Scope != scopeSystem {
		t.Fatalf("got kind=%s scope=%s", ie.Kind, ie.Location.Scope)
	}
}

func TestToolsAccepted(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	tools := []*Tool{{FunctionDeclarations: []*FunctionDeclaration{{Name: "f"}}},
		{CodeExecution: &CodeExecution{}}}
	if _, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(Text("hi"))}, tools); err != nil {
		t.Fatalf("tools should pass: %v", err)
	}
}

// ---- 12. Report never contains bytes and handles are truncated -----------

func TestReportSafety(t *testing.T) {
	c := newTestClient(t)
	longHandle := "files/" + strings.Repeat("a", 40)
	registerFile(c, longHandle,
		"https://generativelanguage.googleapis.com/"+longHandle,
		"image/png", 10, FileStateActive)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	report, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: longHandle})}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(fmt.Sprintf("%v", report)) == 0 {
		t.Fatal("empty report")
	}
	// Now force a rejection with the long handle and confirm truncation.
	withThresholds(t, 5, 16, 1<<30, 64)
	_, err = c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: longHandle})}, nil)
	ie := requireInputError(t, err)
	if len(ie.Handle) >= len(longHandle) {
		t.Errorf("handle was not truncated: %q", ie.Handle)
	}
	if strings.Contains(ie.Error(), "aaaa") && len(ie.Handle) > 27 {
		t.Errorf("full handle leaked into error: %v", ie)
	}
}

// ---- 13. All entries validated; no partial request -----------------------

func TestNoPartialRequest(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	// One good, one bad entry: the whole request is rejected and the report
	// lists the rejected index.
	content := NewUserContent(
		Blob{Data: pngBytes, MIMEType: "image/png"},
		Blob{Data: exeBytes, MIMEType: "application/x-msdownload"},
	)
	report, err := c.validateMultimodalInput(nil, []*Content{content}, nil)
	if err == nil {
		t.Fatal("want error, got nil")
	}
	if len(report.Rejected) != 1 {
		t.Fatalf("rejected = %d, want 1", len(report.Rejected))
	}
	if report.Rejected[0].Location.PartIndex != 1 {
		t.Errorf("rejected part index = %d, want 1", report.Rejected[0].Location.PartIndex)
	}
	// Rejected entries must not appear as accepted stats.
	if st := report.PerModality[ModalityImage]; st.Entries != 1 {
		t.Errorf("accepted image entries = %d, want 1 (bad entry excluded)", st.Entries)
	}
}

// ---- 14. End-to-end through the public request builders ------------------

// countingTransport records whether any request reached the network.
type countingTransport struct {
	calls int
}

func (t *countingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.calls++
	return &http.Response{StatusCode: 200, Body: io.NopCloser(bytes.NewReader(nil))}, nil
}

func TestRejectedRequestNeverSent(t *testing.T) {
	transport := &countingTransport{}
	c, err := NewClient(context.Background(), option.WithAPIKey("k"),
		option.WithEndpoint("https://example.invalid"),
		option.WithHTTPClient(&http.Client{Transport: transport}))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	withThresholds(t, 1<<30, 16, 1<<30, 64)

	m := c.GenerativeModel("test-model")

	// Single-shot.
	if _, err := m.GenerateContent(context.Background(),
		Blob{Data: exeBytes, MIMEType: "application/x-msdownload"}); err == nil {
		t.Fatal("GenerateContent: want error")
	}
	// Streaming.
	iter := m.GenerateContentStream(context.Background(),
		Blob{Data: exeBytes, MIMEType: "application/x-msdownload"})
	if _, err := iter.Next(); err == nil {
		t.Fatal("stream: want error")
	}
	// Chat.
	cs := m.StartChat()
	if _, err := cs.SendMessage(context.Background(),
		Blob{Data: exeBytes, MIMEType: "application/x-msdownload"}); err == nil {
		t.Fatal("chat: want error")
	}
	// Count tokens.
	if _, err := m.CountTokens(context.Background(),
		Blob{Data: exeBytes, MIMEType: "application/x-msdownload"}); err == nil {
		t.Fatal("count tokens: want error")
	}
	// Embeddings.
	em := c.EmbeddingModel("embedding-001")
	if _, err := em.EmbedContent(context.Background(),
		Blob{Data: exeBytes, MIMEType: "application/x-msdownload"}); err == nil {
		t.Fatal("embed: want error")
	}
	batch := em.NewBatch().AddContent(Blob{Data: exeBytes, MIMEType: "application/x-msdownload"})
	if _, err := em.BatchEmbedContents(context.Background(), batch); err == nil {
		t.Fatal("batch embed: want error")
	}
	// Cache creation.
	if _, err := c.CreateCachedContent(context.Background(), &CachedContent{
		Model:    "test-model",
		Contents: []*Content{NewUserContent(Blob{Data: exeBytes, MIMEType: "application/x-msdownload"})},
	}); err == nil {
		t.Fatal("cache: want error")
	}

	if transport.calls != 0 {
		t.Errorf("rejected request reached the network %d times", transport.calls)
	}
}

func TestLastInputReport(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	m := c.GenerativeModel("m")
	_, err := m.newGenerateContentRequest(NewUserContent(Blob{Data: pngBytes}))
	if err != nil {
		t.Fatal(err)
	}
	rep := m.LastInputReport()
	if rep == nil || rep.PerModality[ModalityImage].Entries != 1 {
		t.Fatalf("bad report: %+v", rep)
	}
}

// ---- 15. Serialization compatibility -------------------------------------

func TestSerializationUnchangedForText(t *testing.T) {
	c := newTestClient(t)
	m := c.GenerativeModel("models/x")
	m.SystemInstruction = &Content{Parts: []Part{Text("be brief")}}
	req, err := m.newGenerateContentRequest(NewUserContent(Text("hello"), Text("world")))
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "models/x" {
		t.Errorf("model = %q", req.Model)
	}
	if len(req.Contents) != 1 || len(req.Contents[0].Parts) != 2 {
		t.Fatalf("unexpected contents: %+v", req.Contents)
	}
	// No inline-data field should be synthesized for text.
	for _, p := range req.Contents[0].Parts {
		if _, ok := p.Data.(*pb.Part_InlineData); ok {
			t.Errorf("text part serialized as inline data: %v", p)
		}
	}
	if req.SystemInstruction == nil || req.CachedContent != nil {
		t.Errorf("system/cached fields changed: %+v", req)
	}
}

func TestExplicitMimeSerializationUnchanged(t *testing.T) {
	c := newTestClient(t)
	m := c.GenerativeModel("m")
	req, err := m.newGenerateContentRequest(
		NewUserContent(Blob{Data: pngBytes, MIMEType: "image/png"}))
	if err != nil {
		t.Fatal(err)
	}
	blob := req.Contents[0].Parts[0].GetInlineData()
	if blob.MimeType != "image/png" || !bytes.Equal(blob.Data, pngBytes) {
		t.Errorf("blob changed: %+v", blob)
	}
}

// ---- 16. Error unwrapping and AsInputError -------------------------------

func TestErrorUnwrapAndHelper(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: "/bad"})}, nil)
	if AsInputError(err) == nil {
		t.Fatal("AsInputError returned nil")
	}
	var ie *InputError
	if !errors.As(err, &ie) {
		t.Fatal("errors.As failed")
	}
	if ie.Unwrap() != nil {
		t.Errorf("expected nil wrapped error, got %v", ie.Unwrap())
	}
}

// ---- 17. Equivalence of error category across the two paths --------------

func TestInlineAndReferenceSameErrorCategory(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "application/x-msdownload", int64(len(exeBytes)), FileStateActive)
	withThresholds(t, 1<<30, 16, 1<<30, 64)

	_, inlineErr := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(Blob{Data: exeBytes, MIMEType: "application/x-msdownload"})}, nil)
	_, refErr := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: testHandle})}, nil)
	ie1 := requireInputError(t, inlineErr)
	ie2 := requireInputError(t, refErr)
	if ie1.Kind != ie2.Kind {
		t.Errorf("categories differ: inline=%s reference=%s", ie1.Kind, ie2.Kind)
	}
	if ie1.Modality != ie2.Modality {
		t.Errorf("modalities differ: inline=%s reference=%s", ie1.Modality, ie2.Modality)
	}
}

// ---- helpers --------------------------------------------------------------

func requireInputError(t *testing.T, err error) *InputError {
	t.Helper()
	if err == nil {
		t.Fatal("want error, got nil")
	}
	ie := AsInputError(err)
	if ie == nil {
		t.Fatalf("error %v (%T) is not *InputError", err, err)
	}
	return ie
}

func assertModalityStats(t *testing.T, r *InputReport, modality InputModality, entries int, bytes int64) {
	t.Helper()
	st := r.PerModality[modality]
	if st.Entries != entries || st.Bytes != bytes {
		t.Errorf("stats for %s = %+v, want entries=%d bytes=%d", modality, st, entries, bytes)
	}
}

// ---- 18. Batch embedding inference write-back ----------------------------

func TestBatchEmbedInferenceWritesBack(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	em := c.EmbeddingModel("embedding-001")
	batch := em.NewBatch().AddContent(Blob{Data: pngBytes})
	contents := make([]*Content, len(batch.entries))
	for i, e := range batch.entries {
		contents[i] = e.content
	}
	if _, err := c.validateMultimodalInput(nil, contents, nil); err != nil {
		t.Fatal(err)
	}
	req := batch.toProto()
	if got := req.Requests[0].Content.Parts[0].GetInlineData().MimeType; got != "image/png" {
		t.Errorf("batch inline MIME = %q, want image/png", got)
	}
}

// ---- 19. Golden serialization: legal request is field-for-field identical --

func TestGoldenTextRequestSerialization(t *testing.T) {
	c := newTestClient(t)
	m := c.GenerativeModel("models/golden")
	m.SetTemperature(0.25)
	m.SetMaxOutputTokens(7)
	m.SystemInstruction = &Content{Role: roleUser, Parts: []Part{Text("sys")}}
	got, err := m.newGenerateContentRequest(NewUserContent(Text("a"), Text("b")))
	if err != nil {
		t.Fatal(err)
	}
	want := &pb.GenerateContentRequest{
		Model: "models/golden",
		Contents: []*pb.Content{{
			Role:  roleUser,
			Parts: []*pb.Part{{Data: &pb.Part_Text{Text: "a"}}, {Data: &pb.Part_Text{Text: "b"}}},
		}},
		SystemInstruction: &pb.Content{
			Role:  roleUser,
			Parts: []*pb.Part{{Data: &pb.Part_Text{Text: "sys"}}},
		},
		GenerationConfig: &pb.GenerationConfig{
			Temperature:     ptrFloat32(0.25),
			MaxOutputTokens: ptrInt32(7),
		},
	}
	if !protoEqual(got, want) {
		t.Fatalf("serialization drift:\n got=%v\nwant=%v", got, want)
	}
}

func TestGoldenExplicitBlobSerialization(t *testing.T) {
	c := newTestClient(t)
	m := c.GenerativeModel("models/golden")
	got, err := m.newGenerateContentRequest(NewUserContent(
		Text("caption"),
		Blob{MIMEType: "image/png", Data: pngBytes},
	))
	if err != nil {
		t.Fatal(err)
	}
	want := &pb.GenerateContentRequest{
		Model: "models/golden",
		Contents: []*pb.Content{{
			Role: roleUser,
			Parts: []*pb.Part{
				{Data: &pb.Part_Text{Text: "caption"}},
				{Data: &pb.Part_InlineData{InlineData: &pb.Blob{MimeType: "image/png", Data: pngBytes}}},
			},
		}},
		// The veneer always emits an (empty) GenerationConfig; that is
		// pre-existing behavior and must remain unchanged.
		GenerationConfig: &pb.GenerationConfig{},
	}
	if !protoEqual(got, want) {
		t.Fatalf("serialization drift:\n got=%v\nwant=%v", got, want)
	}
}

// ---- 20. History error locates the exact historical entry ----------------

func TestHistoryErrorPinpointsEntry(t *testing.T) {
	c := newTestClient(t)
	withThresholds(t, 1<<30, 16, 1<<30, 64)
	history := []*Content{
		NewUserContent(Text("turn 1"), Blob{Data: pngBytes, MIMEType: "image/png"}),
		NewUserContent(Text("turn 2")),
		NewUserContent(Text("turn 3"), FileData{URI: "/local/not/allowed"}),
	}
	_, err := c.validateMultimodalInput(nil, history, nil)
	ie := requireInputError(t, err)
	if ie.Kind != InputErrorMalformed {
		t.Fatalf("kind = %s", ie.Kind)
	}
	if ie.Location.ContentIndex != 2 || ie.Location.PartIndex != 1 {
		t.Fatalf("location = %+v, want contents[2].parts[1]", ie.Location)
	}
}

// ---- 21. Processing retry guidance distinct from terminal states ---------

func TestProcessingMessageGuidance(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "image/png", 10, FileStateProcessing)
	_, err := c.validateMultimodalInput(nil,
		[]*Content{NewUserContent(FileData{URI: testHandle})}, nil)
	ie := requireInputError(t, err)
	if !strings.Contains(strings.ToLower(ie.Error()), "getfile") {
		t.Errorf("processing error should suggest GetFile: %v", ie)
	}
}

func ptrFloat32(f float32) *float32 { return &f }
func ptrInt32(i int32) *int32       { return &i }

func protoEqual(a, b proto.Message) bool { return proto.Equal(a, b) }

// ---- 22. Repeated references count entries even when bytes dedup --------

func TestRepeatedReferenceEntriesCounted(t *testing.T) {
	c := newTestClient(t)
	registerFile(c, testHandle, testURI, "image/png", 100, FileStateActive)
	withThresholds(t, 1<<30, 2, 1<<30, 64) // only 2 image entries allowed
	parts := []Part{FileData{URI: testHandle}, FileData{URI: testHandle}, FileData{URI: testHandle}}
	_, err := c.validateMultimodalInput(nil, []*Content{NewUserContent(parts...)}, nil)
	ie := requireInputError(t, err)
	if ie.BudgetItem != BudgetItemPerModality || ie.Metric != MetricCount {
		t.Fatalf("got item=%s metric=%s, want per_modality/entries", ie.BudgetItem, ie.Metric)
	}
	if ie.OverBy != 1 {
		t.Errorf("overby = %d, want 1 (3 entries, limit 2)", ie.OverBy)
	}
}
