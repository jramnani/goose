import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import GooseMessage from './GooseMessage';
import { IntlTestWrapper } from '../i18n/test-utils';
import type { Message, MessageContent } from '../types/message';

function makeAssistantMessage(content: MessageContent[]): Message {
  return {
    content,
    created: Date.now(),
    metadata: { agentVisible: true, userVisible: true },
    role: 'assistant',
  };
}

function renderGooseMessage(message: Message, isStreaming = false) {
  return render(
    <GooseMessage
      sessionId="session-1"
      message={message}
      hideTimestamp={false}
      toolStates={[]}
      toolNotifications={[]}
      toolConfirmationShownInline={false}
      append={() => {}}
      isStreaming={isStreaming}
    />,
    { wrapper: IntlTestWrapper }
  );
}

describe('GooseMessage copy button', () => {
  it('shows the copy button on a finished message that has a thinking block', () => {
    const message = makeAssistantMessage([
      { type: 'thinking', thinking: 'The user wants TypeScript highlights.', signature: 'sig' },
      { type: 'text', text: 'Here are some highlights of the TypeScript language.' },
    ]);

    renderGooseMessage(message);

    expect(screen.getByText('Copy')).toBeInTheDocument();
  });

  it('hides the copy button while the message is streaming', () => {
    const message = makeAssistantMessage([
      { type: 'thinking', thinking: 'The user wants TypeScript highlights.', signature: 'sig' },
      { type: 'text', text: 'Here are some highlights of the TypeScript language.' },
    ]);

    renderGooseMessage(message, true);

    expect(screen.queryByText('Copy')).not.toBeInTheDocument();
  });
});
