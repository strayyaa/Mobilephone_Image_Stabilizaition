import { useState } from "react";
import { uploadFile } from "@/mock/api";
import type { FilePurpose, UploadedFileMeta } from "@/types/contracts";

export function useFileUpload() {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string>();

  async function upload(file: File, purpose: FilePurpose): Promise<UploadedFileMeta> {
    setUploading(true);
    setError(undefined);
    try {
      return await uploadFile({ file, purpose });
    } catch (err) {
      const message = err instanceof Error ? err.message : "上传失败";
      setError(message);
      throw err;
    } finally {
      setUploading(false);
    }
  }

  return {
    uploading,
    error,
    upload,
  };
}
