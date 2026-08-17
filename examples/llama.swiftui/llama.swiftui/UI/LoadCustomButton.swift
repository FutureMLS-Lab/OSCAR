import SwiftUI
import UniformTypeIdentifiers

struct LoadCustomButton: View {
    @ObservedObject private var llamaState: LlamaState
    @State private var showFileImporter = false

    init(llamaState: LlamaState) {
        self.llamaState = llamaState
    }

    var body: some View {
        VStack {
            Button(action: {
                showFileImporter = true
            }) {
                Label("Import GGUF from Files", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.borderedProminent)
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [UTType(filenameExtension: "gguf", conformingTo: .data)!],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let files):
                files.forEach { file in
                    Task {
                        let gotAccess = file.startAccessingSecurityScopedResource()
                        if !gotAccess { return }
                        defer { file.stopAccessingSecurityScopedResource() }

                        await llamaState.importModel(from: file.absoluteURL)
                    }
                }
            case .failure(let error):
                print(error)
            }
        }
    }
}
