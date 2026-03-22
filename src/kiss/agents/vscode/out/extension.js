"use strict";
/**
 * KISS Sorcar VS Code Extension entry point.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const SorcarPanel_1 = require("./SorcarPanel");
const MergeManager_1 = require("./MergeManager");
let primaryProvider;
let secondaryProvider;
let mergeManager;
function getActiveProvider() {
    return secondaryProvider ?? primaryProvider;
}
function getActiveMergeManager() {
    return getActiveProvider()?.mergeManager ?? mergeManager;
}
function activate(context) {
    console.log('KISS Sorcar extension activating...');
    // Check if VS Code supports secondary sidebar (1.98+)
    const supportsSecondarySidebar = typeof vscode.ViewColumn !== 'undefined';
    mergeManager = new MergeManager_1.MergeManager();
    context.subscriptions.push({ dispose: () => mergeManager?.dispose() });
    // Create and register the primary (activitybar) webview provider
    primaryProvider = new SorcarPanel_1.SorcarViewProvider(context.extensionUri, mergeManager);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider('kissSorcar.chatView', primaryProvider, { webviewOptions: { retainContextWhenHidden: true } }));
    // Create and register the secondary sidebar webview provider
    secondaryProvider = new SorcarPanel_1.SorcarViewProvider(context.extensionUri, mergeManager);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider('kissSorcar.chatViewSecondary', secondaryProvider, { webviewOptions: { retainContextWhenHidden: true } }));
    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.openPanel', () => {
        vscode.commands.executeCommand('kissSorcar.chatViewSecondary.focus').then(undefined, () => vscode.commands.executeCommand('kissSorcar.chatView.focus'));
    }));
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.newConversation', () => {
        getActiveProvider()?.newConversation();
    }));
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.stopTask', () => {
        getActiveProvider()?.stopTask();
    }));
    // Merge commands
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.acceptChange', () => {
        getActiveMergeManager()?.acceptChange();
    }), vscode.commands.registerCommand('kissSorcar.rejectChange', () => {
        getActiveMergeManager()?.rejectChange();
    }), vscode.commands.registerCommand('kissSorcar.prevChange', () => {
        getActiveMergeManager()?.prevChange();
    }), vscode.commands.registerCommand('kissSorcar.nextChange', () => {
        getActiveMergeManager()?.nextChange();
    }), vscode.commands.registerCommand('kissSorcar.acceptAll', () => {
        getActiveMergeManager()?.acceptAll();
    }), vscode.commands.registerCommand('kissSorcar.rejectAll', () => {
        getActiveMergeManager()?.rejectAll();
    }));
    // Set context for conditional view visibility, then auto-open on startup
    vscode.commands.executeCommand('setContext', 'kissSorcar:doesNotSupportSecondarySidebar', !supportsSecondarySidebar).then(() => {
        vscode.commands.executeCommand('kissSorcar.openPanel');
    });
    console.log('KISS Sorcar extension activated');
}
function deactivate() {
    primaryProvider?.dispose();
    primaryProvider = undefined;
    secondaryProvider?.dispose();
    secondaryProvider = undefined;
    console.log('KISS Sorcar extension deactivated');
}
//# sourceMappingURL=extension.js.map