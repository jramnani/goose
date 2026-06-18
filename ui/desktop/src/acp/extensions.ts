import type { ExtensionResponse, ExtensionEntry } from '../api';
import type { GooseExtensionEntry, McpServer } from '@aaif/goose-sdk';
import { getAcpClient } from './acpConnection';

function headersToRecord(headers: { name: string; value: string }[] = []) {
  return Object.fromEntries(headers.map(({ name, value }) => [name, value]));
}

// The ACP `McpServer` types expose literal environment variables as an
// `EnvVariable[]` (each `{ name, value }`). The Goose `ExtensionConfig` shape
// instead stores them as an `envs` record (`{ [name]: value }`). Convert
// between the two so that literal `envs` configured in `config.yaml` are
// preserved when the desktop app forwards extensions to the server as
// `extension_overrides`. Without this, literal env values are silently dropped
// (only `env_keys` survived), so stdio subprocesses launched without their
// configured environment and fell back to their own defaults.
function envVariablesToRecord(
  environmentVariables: { name: string; value: string }[] = []
): Record<string, string> {
  return Object.fromEntries(
    environmentVariables.map(({ name, value }) => [name, value])
  );
}

function mcpServerToExtension(
  server: McpServer,
  entry: GooseExtensionEntry
): ExtensionEntry | null {
  const extension = entry.extension;
  if (extension.type !== 'mcp') {
    return null;
  }

  if ('command' in server) {
    return {
      type: 'stdio',
      enabled: entry.enabled,
      name: server.name,
      description: extension.description ?? '',
      cmd: server.command,
      args: server.args,
      envs: envVariablesToRecord(server.env),
      env_keys: extension.envKeys ?? [],
      timeout: extension.timeout,
      bundled: extension.bundled,
    };
  }

  if ('url' in server) {
    return {
      type: 'streamable_http',
      enabled: entry.enabled,
      name: server.name,
      description: extension.description ?? '',
      uri: server.url,
      headers: headersToRecord(server.headers),
      env_keys: extension.envKeys ?? [],
      timeout: extension.timeout,
      socket: extension.socket,
      bundled: extension.bundled,
    };
  }

  return null;
}

function gooseExtensionEntryToExtensionEntry(entry: GooseExtensionEntry): ExtensionEntry | null {
  const extension = entry.extension;

  switch (extension.type) {
    case 'builtin':
    case 'platform':
      return {
        ...extension,
        description: extension.description ?? '',
        enabled: entry.enabled,
      };
    case 'mcp':
      return mcpServerToExtension(extension.server, entry);
  }

  return null;
}

export async function getConfiguredExtensions(): Promise<ExtensionResponse> {
  const client = await getAcpClient();
  const response = await client.goose.configExtensionsList_unstable({});
  return {
    extensions: response.extensions
      .map(gooseExtensionEntryToExtensionEntry)
      .filter((entry): entry is ExtensionEntry => entry !== null),
    warnings: response.warnings ?? [],
  };
}
