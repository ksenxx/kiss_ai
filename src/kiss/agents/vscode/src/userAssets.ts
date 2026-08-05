// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export function kissHomeDir(): string {
  return process.env.KISS_HOME || path.join(os.homedir(), '.kiss');
}

export function ensureUserAsset(name: string, packagePath: string): string {
  const userPath = path.join(kissHomeDir(), name);
  try {
    if (fs.existsSync(userPath)) {
      return userPath;
    }
    if (fs.existsSync(packagePath)) {
      fs.mkdirSync(path.dirname(userPath), {recursive: true});
      fs.copyFileSync(packagePath, userPath);
      return userPath;
    }
  } catch {}
  return packagePath;
}

export function ensureUserAssetFromDefault(
  name: string,
  defaultContent: string,
): string | null {
  const userPath = path.join(kissHomeDir(), name);
  try {
    if (fs.existsSync(userPath)) return userPath;
    fs.mkdirSync(path.dirname(userPath), {recursive: true});
    fs.writeFileSync(userPath, defaultContent);
    return userPath;
  } catch {
    return null;
  }
}
