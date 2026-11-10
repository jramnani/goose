import { describe, it, expect } from 'vitest';
import { matchedSnippetSegments } from '../components/sessions/SessionListView';

describe('matchedSnippetSegments', () => {
  it('matches terms case-insensitively', () => {
    expect(matchedSnippetSegments('fix PROJ-1234 today', ['proj-1234'])).toEqual([
      { text: 'fix ', highlighted: false },
      { text: 'PROJ-1234', highlighted: true },
      { text: ' today', highlighted: false },
    ]);
  });

  it('highlights any of several OR terms', () => {
    expect(matchedSnippetSegments('migrated the SQLite file', ['postgres', 'sqlite'])).toEqual([
      { text: 'migrated the ', highlighted: false },
      { text: 'SQLite', highlighted: true },
      { text: ' file', highlighted: false },
    ]);
  });

  it('highlights a term at the start and end of the snippet', () => {
    expect(matchedSnippetSegments('alpha omega', ['alpha'])).toEqual([
      { text: 'alpha', highlighted: true },
      { text: ' omega', highlighted: false },
    ]);
    expect(matchedSnippetSegments('alpha omega', ['omega'])).toEqual([
      { text: 'alpha ', highlighted: false },
      { text: 'omega', highlighted: true },
    ]);
  });

  it('highlights every occurrence of a term', () => {
    expect(matchedSnippetSegments('again and again', ['again'])).toEqual([
      { text: 'again', highlighted: true },
      { text: ' and ', highlighted: false },
      { text: 'again', highlighted: true },
    ]);
  });

  it('returns a single unhighlighted segment for empty or blank terms', () => {
    expect(matchedSnippetSegments('plain text', [])).toEqual([
      { text: 'plain text', highlighted: false },
    ]);
    expect(matchedSnippetSegments('plain text', ['   '])).toEqual([
      { text: 'plain text', highlighted: false },
    ]);
  });

  it('returns a single plain segment when no term occurs in the snippet', () => {
    expect(matchedSnippetSegments('nothing here', ['postgres'])).toEqual([
      { text: 'nothing here', highlighted: false },
    ]);
  });
});