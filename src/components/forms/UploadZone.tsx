import { useRef, useState } from "react";
import { CheckCircle2, Loader2, UploadCloud } from "lucide-react";
import { useFileUpload } from "@/features/uploads/useFileUpload";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { FilePurpose, UploadedFileMeta } from "@/types/contracts";

interface UploadZoneProps {
  title: string;
  hint: string;
  accept: string;
  purpose: FilePurpose;
  required?: boolean;
  value?: UploadedFileMeta;
  onChange: (file: UploadedFileMeta) => void;
}

export function UploadZone({ title, hint, accept, purpose, required, value, onChange }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const { upload, uploading, error } = useFileUpload();

  async function handleFile(selected: File | undefined) {
    if (!selected) return;
    try {
      const result = await upload(selected, purpose);
      onChange(result);
    } catch {}
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">
          {title}
          {required ? <span className="ml-1 text-rose-300">*</span> : null}
        </label>
        <span className="text-xs text-muted-foreground">{hint}</span>
      </div>
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void handleFile(event.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "cursor-pointer rounded-xl border border-dashed p-5 transition-all",
          dragging ? "border-primary bg-primary/10" : "border-border bg-card/40 hover:bg-card/60"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(event) => void handleFile(event.target.files?.[0])}
        />
        <div className="flex items-center gap-3">
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
          ) : value ? (
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
          ) : (
            <UploadCloud className="h-5 w-5 text-muted-foreground" />
          )}
          <div>
            {value ? (
              <div className="space-y-1">
                <p className="text-sm font-medium">{value.fileName}</p>
                <p className="text-xs text-muted-foreground">
                  {formatBytes(value.size)} · 已上传 · fileId={value.fileId}
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-sm">点击或拖拽上传文件</p>
                <p className="text-xs text-muted-foreground">支持格式：{accept}</p>
              </div>
            )}
          </div>
        </div>
      </div>
      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  );
}
