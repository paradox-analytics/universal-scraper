import { useState } from 'react';

interface FieldSelectorProps {
  fields: string[];
  onChange: (fields: string[]) => void;
  mode?: 'natural' | 'structured';
}

export function FieldSelector({ fields, onChange, mode = 'natural' }: FieldSelectorProps) {
  const [naturalLanguage, setNaturalLanguage] = useState('');
  const [structuredFields, setStructuredFields] = useState<string[]>(fields);
  const [currentMode, setCurrentMode] = useState<'natural' | 'structured'>(mode);

  const handleNaturalLanguageChange = (value: string) => {
    setNaturalLanguage(value);
    // Parse natural language into fields (simple implementation)
    if (value) {
      const extracted = value
        .split(/[,\n]/)
        .map(f => f.trim())
        .filter(f => f.length > 0);
      onChange(extracted);
    } else {
      onChange([]);
    }
  };

  const handleAddField = () => {
    const newField = prompt('Enter field name:');
    if (newField && !structuredFields.includes(newField)) {
      const updated = [...structuredFields, newField];
      setStructuredFields(updated);
      onChange(updated);
    }
  };

  const handleRemoveField = (field: string) => {
    const updated = structuredFields.filter(f => f !== field);
    setStructuredFields(updated);
    onChange(updated);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="block text-sm font-medium text-gray-700">
          Fields to Extract
        </label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setCurrentMode('natural')}
            className={`text-xs px-2 py-1 rounded ${
              currentMode === 'natural'
                ? 'bg-primary-100 text-primary-700'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            Natural Language
          </button>
          <button
            type="button"
            onClick={() => setCurrentMode('structured')}
            className={`text-xs px-2 py-1 rounded ${
              currentMode === 'structured'
                ? 'bg-primary-100 text-primary-700'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            Structured
          </button>
        </div>
      </div>

      {currentMode === 'natural' ? (
        <div>
          <textarea
            value={naturalLanguage}
            onChange={(e) => handleNaturalLanguageChange(e.target.value)}
            placeholder="e.g., Extract product names, prices, ratings, and descriptions"
            className="input-field h-24 resize-none"
          />
          <p className="mt-1 text-xs text-gray-500">
            Describe what you want to extract in plain English
          </p>
          {fields.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {fields.map((field) => (
                <span
                  key={field}
                  className="inline-flex items-center px-2 py-1 rounded-md bg-primary-100 text-primary-800 text-sm"
                >
                  {field}
                </span>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div>
          <div className="flex gap-2 mb-2">
            <input
              type="text"
              placeholder="Add field name"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  const input = e.currentTarget;
                  const value = input.value.trim();
                  if (value && !structuredFields.includes(value)) {
                    const updated = [...structuredFields, value];
                    setStructuredFields(updated);
                    onChange(updated);
                    input.value = '';
                  }
                }
              }}
              className="input-field flex-1"
            />
            <button
              type="button"
              onClick={handleAddField}
              className="btn-secondary whitespace-nowrap"
            >
              Add Field
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {structuredFields.map((field) => (
              <span
                key={field}
                className="inline-flex items-center px-3 py-1 rounded-md bg-gray-100 text-gray-800 text-sm"
              >
                {field}
                <button
                  type="button"
                  onClick={() => handleRemoveField(field)}
                  className="ml-2 text-gray-500 hover:text-gray-700"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

