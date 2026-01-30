import { useState, useRef } from 'react';
import { DocumentArrowUpIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedTypes?: string;
  maxSizeMB?: number;
}

export function FileUpload({ onFileSelect, acceptedTypes = '.pdf,.doc,.docx,.txt,.md', maxSizeMB = 100 }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // Validate file type
    const extension = selectedFile.name.split('.').pop()?.toLowerCase();
    const allowedExtensions = acceptedTypes.split(',').map(ext => ext.replace('.', '').trim());
    
    if (extension && !allowedExtensions.includes(extension)) {
      setError(`File type not supported. Allowed types: ${acceptedTypes}`);
      return;
    }

    // Validate file size
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (selectedFile.size > maxSizeBytes) {
      setError(`File size exceeds ${maxSizeMB}MB limit`);
      return;
    }

    setFile(selectedFile);
    setError('');
    onFileSelect(selectedFile);
  };

  const handleRemove = () => {
    setFile(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div>
      <label className="label">Upload Document</label>
      
      {!file ? (
        <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-700 border-dashed rounded-xl hover:border-purple-500 transition-colors bg-gray-900 cursor-pointer" onClick={() => fileInputRef.current?.click()}>
          <div className="space-y-1 text-center">
            <DocumentArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
            <div className="flex text-sm text-gray-300">
              <label htmlFor="file-upload" className="relative cursor-pointer font-medium text-purple-400 hover:text-purple-300 focus-within:outline-none">
                <span>Upload a file</span>
                <input
                  id="file-upload"
                  ref={fileInputRef}
                  name="file-upload"
                  type="file"
                  className="sr-only"
                  accept={acceptedTypes}
                  onChange={handleFileChange}
                />
              </label>
              <p className="pl-1">or drag and drop</p>
            </div>
            <p className="text-xs text-gray-400">
              {acceptedTypes} up to {maxSizeMB}MB
            </p>
          </div>
        </div>
      ) : (
        <div className="mt-1 flex items-center justify-between p-4 bg-gray-800 rounded-xl border border-gray-700">
          <div className="flex items-center space-x-3">
            <DocumentArrowUpIcon className="h-8 w-8 text-purple-400" />
            <div>
              <p className="text-sm font-medium text-gray-100">{file.name}</p>
              <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleRemove}
            className="text-gray-400 hover:text-gray-200"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
      )}

      {error && (
        <p className="mt-2 text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}

