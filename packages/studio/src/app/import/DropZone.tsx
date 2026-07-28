/*
React component — whole-window drag-and-drop import. Wraps the app; dragging
a CAD file anywhere over the window shows the dashed overlay, dropping feeds
the file into the same import pipeline as the picker. Unsupported files are
refused with an error in the store (shown by the status bar), not silently.
*/

import { useEffect, useState, type ReactNode } from 'react';
import { importModelFile } from '../state/modelSession';
import { isSupportedCadFile, SUPPORTED_EXTENSIONS } from '../../engine/model/loadModel';
import { setState } from '../state/store';

export function DropZone({ children }: { children: ReactNode }) {
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    // Depth counter, not a boolean: dragenter/dragleave also fire on every
    // child element the file passes over.
    let dragDepth = 0;

    function onDragEnter(event: DragEvent) {
      if (!event.dataTransfer?.types.includes('Files')) return;
      dragDepth += 1;
      setDragActive(true);
    }
    function onDragLeave() {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) setDragActive(false);
    }
    function onDragOver(event: DragEvent) {
      event.preventDefault(); // required, or the browser navigates to the file
    }
    function onDrop(event: DragEvent) {
      event.preventDefault();
      dragDepth = 0;
      setDragActive(false);
      const file = event.dataTransfer?.files?.[0];
      if (!file) return;
      if (!isSupportedCadFile(file.name)) {
        setState({
          importPhase: 'error',
          importError: `"${file.name}" is not a supported CAD file (${SUPPORTED_EXTENSIONS.join(', ')}).`,
        });
        return;
      }
      void importModelFile(file).catch(() => {});
    }

    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('drop', onDrop);
    };
  }, []);

  return (
    <>
      {children}
      {dragActive && (
        <div className="drop-overlay">
          <span className="drop-overlay__label">
            Drop to import ({SUPPORTED_EXTENSIONS.map((e) => `.${e}`).join(' ')})
          </span>
        </div>
      )}
    </>
  );
}
