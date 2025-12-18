import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';
import { ClipboardIcon, CheckIcon } from '@heroicons/react/24/outline';

interface CodeViewerProps {
  data: any;
  language?: 'json' | 'html' | 'javascript' | 'python';
  title?: string;
}

export function CodeViewer({ data, language = 'json', title = 'Raw Data' }: CodeViewerProps) {
  const [copied, setCopied] = useState(false);

  const codeString =
    language === 'json'
      ? JSON.stringify(data, null, 2)
      : typeof data === 'string'
      ? data
      : JSON.stringify(data, null, 2);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="bg-gray-800 px-4 py-2 flex justify-between items-center">
        <span className="text-white text-sm font-medium">{title}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-2 text-white text-sm hover:text-gray-300 transition-colors"
        >
          {copied ? (
            <>
              <CheckIcon className="h-4 w-4" />
              Copied!
            </>
          ) : (
            <>
              <ClipboardIcon className="h-4 w-4" />
              Copy
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        customStyle={{ margin: 0, borderRadius: 0 }}
        showLineNumbers
      >
        {codeString}
      </SyntaxHighlighter>
    </div>
  );
}

