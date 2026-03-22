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
let sorcarProvider;
function activate(context) {
    console.log('KISS Sorcar extension activating...');
    // Create the webview provider
    sorcarProvider = new SorcarPanel_1.SorcarViewProvider(context.extensionUri);
    // Register the webview provider for the sidebar view
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(SorcarPanel_1.SorcarViewProvider.viewType, sorcarProvider, {
        webviewOptions: {
            retainContextWhenHidden: true,
        },
    }));
    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.openPanel', () => {
        // Focus on the sorcar view
        vscode.commands.executeCommand('kissSorcar.chatView.focus');
    }));
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.newConversation', () => {
        if (sorcarProvider) {
            sorcarProvider.newConversation();
        }
    }));
    context.subscriptions.push(vscode.commands.registerCommand('kissSorcar.stopTask', () => {
        if (sorcarProvider) {
            sorcarProvider.stopTask();
        }
    }));
    console.log('KISS Sorcar extension activated');
}
function deactivate() {
    if (sorcarProvider) {
        sorcarProvider.dispose();
        sorcarProvider = undefined;
    }
    console.log('KISS Sorcar extension deactivated');
}
//# sourceMappingURL=extension.js.map