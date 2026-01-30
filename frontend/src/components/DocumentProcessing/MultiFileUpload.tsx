import { useState, useRef } from 'react';
import { DocumentArrowUpIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface MultiFileUploadProps {
  onFilesSelect: (files: File[]) => void;
  acceptedTypes?: string;
  maxSizeMB?: number;
}

export function MultiFileUpload({ onFilesSelect, acceptedTypes = '.pdf,.doc,.docx,.txt,.md', maxSizeMB = 100 }: MultiFileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const validFiles: File[] = [];
    const errors: string[] = [];

    selectedFiles.forEach((file) => {
      // Validate file type
      const extension = file.name.split('.').pop()?.toLowerCase();
      const allowedExtensions = acceptedTypes.split(',').map(ext => ext.replace('.', '').trim());
      
      if (extension && !allowedExtensions.includes(extension)) {
        errors.push(`${file.name}: File type not supported`);
        return;
      }

      // Validate file size
      const maxSizeBytes = maxSizeMB * 1024 * 1024;
      if (file.size > maxSizeBytes) {
        errors.push(`${file.name}: File size exceeds ${maxSizeMB}MB limit`);
        return;
      }

      validFiles.push(file);
    });

    if (errors.length > 0) {
      setError(errors.join('; '));
    } else {
      setError('');
    }

    if (validFiles.length > 0) {
      const updatedFiles = [...files, ...validFiles];
      setFiles(updatedFiles);
      onFilesSelect(updatedFiles);
    }
  };

  const handleRemove = (index: number) => {
    const updatedFiles = files.filter((_, i) => i !== index);
    setFiles(updatedFiles);
    onFilesSelect(updatedFiles);
    setError('');
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div>
      <label className="label">Upload Documents</label>
      
      <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-700 border-dashed rounded-xl hover:border-purple-500 transition-colors bg-gray-900 cursor-pointer" onClick={() => fileInputRef.current?.click()}>
        <div className="space-y-1 text-center">
          <DocumentArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
          <div className="flex text-sm text-gray-300">
            <label htmlFor="file-upload" className="relative cursor-pointer font-medium text-purple-400 hover:text-purple-300 focus-within:outline-none">
              <span>Upload files</span>
              <input
                id="file-upload"
                ref={fileInputRef}
                name="file-upload"
                type="file"
                className="sr-only"
                accept={acceptedTypes}
                multiple
                onChange={handleFileChange}
              />
            </label>
            <p className="pl-1">or drag and drop</p>
          </div>
          <p className="text-xs text-gray-400">
            {acceptedTypes} up to {maxSizeMB}MB each
          </p>
        </div>
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-2 max-h-60 overflow-y-auto">
          {files.map((file, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-800 rounded-xl border border-gray-700">
              <div className="flex items-center space-x-3 flex-1 min-w-0">
                <DocumentArrowUpIcon className="h-6 w-6 text-purple-400 flex-shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-gray-100 truncate">{file.name}</p>
                  <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => handleRemove(index)}
                className="text-gray-400 hover:text-gray-200 flex-shrink-0 ml-2"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {error && (
        <p className="mt-2 text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}




