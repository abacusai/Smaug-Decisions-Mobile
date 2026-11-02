import SwiftUI

@main
struct CascadeApp: App {
    @UIApplicationDelegateAdaptor(CascadeAppDelegate.self) private var delegate
    @StateObject private var model = CascadeModel()
    var body: some Scene {
        WindowGroup { RootView().environmentObject(model).environmentObject(ModelDownloader.shared) }
    }
}

/// The first-run setup screen once (FirstRun), then the home screen.
struct RootView: View {
    @State private var firstRun = FirstRun.shouldShowNow()
    var body: some View {
        if firstRun {
            FirstRunView { withAnimation(.easeInOut(duration: 0.2)) { firstRun = false } }.transition(.opacity)
        } else {
            ContentView()
        }
    }
}
