"use client";
import { useState } from "react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null); // Store API response

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) validateFile(file);
  };

  const validateFile = (file: File) => {
    const allowedTypes = ["image/jpeg", "image/png", "video/mp4", "video/quicktime"];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!allowedTypes.includes(file.type)) {
      setError("Only JPG, PNG images and MP4, MOV videos are allowed.");
      setSelectedFile(null);
    } else if (file.size > maxSize) {
      setError("File size should be less than 50MB.");
      setSelectedFile(null);
    } else {
      setError(null);
      setSelectedFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    setResult(null);
    
    const formData = new FormData();
    formData.append("file", selectedFile);

    // try {
    //   const response = await fetch("http://localhost:5000/analyze", {  // Replace with your API URL
    //     method: "POST",
    //     body: formData,
    //   });

    //   if (!response.ok) {
    //     throw new Error("Failed to analyze the file.");
    //   }

    //   const data = await response.json();
    //   setResult(data.result); // Assuming the API returns { "result": "Real" or "Fake" }
    // } catch (err) {
    //   setError("Error analyzing file. Please try again.");
    // } finally {
    //   setIsLoading(false);
    // }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <h1 className="text-4xl font-bold text-gray-900 mb-6">
        Deepfake Detector
      </h1>

      <div className="bg-white p-6 rounded-xl shadow-xl w-full max-w-lg">
        {/* Upload Area */}
        <label className="block w-full cursor-pointer border-2 border-dashed border-gray-300 bg-gray-50 text-gray-600 rounded-lg p-6 text-center hover:bg-gray-100 transition">
          <input
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <p className="text-gray-600">
            {selectedFile ? (
              <span className="text-green-600 font-semibold">
                File Selected!
              </span>
            ) : (
              <>
                Drag & Drop or{" "}
                <span className="text-gray-700 font-semibold">
                  Click to Upload
                </span>
              </>
            )}
          </p>
          <p className="text-xs text-gray-500">
            JPG, PNG, MP4, MOV (Max: 50MB)
          </p>
        </label>

        {error && <p className="text-red-500 text-sm mt-2">{error}</p>}

        {/* File Preview */}
        {selectedFile && (
          <div className="mt-4 p-4 border rounded-lg bg-gray-100 flex flex-col items-center">
            {selectedFile.type.startsWith("image/") ? (
              <img
                src={URL.createObjectURL(selectedFile)}
                alt="preview"
                className="w-32 h-32 rounded-md object-cover"
              />
            ) : selectedFile.type.startsWith("video/") ? (
              <video
                src={URL.createObjectURL(selectedFile)}
                controls
                className="w-[300px] h-auto rounded-lg"
              />
            ) : null}

            <div className="mt-2 text-center">
              <p className="text-gray-700 font-medium">{selectedFile.name}</p>
              <p className="text-sm text-gray-600">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        )}

        {/* Analyze Button */}
        {selectedFile && (
          <div className="mt-6">
            <button
              onClick={handleAnalyze}
              disabled={isLoading}
              className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition ${
                isLoading
                  ? "bg-blue-400 cursor-not-allowed"
                  : "bg-blue-700 hover:bg-blue-800 text-white border-2 border-blue-800 shadow-md"
              }`}
            >
              {isLoading ? (
                <div className="flex justify-center items-center">
                  <div className="w-5 h-5 border-4 border-t-transparent border-blue-300 rounded-full animate-spin"></div>
                </div>
              ) : (
                "Analyze File"
              )}
            </button>
          </div>
        )}

        {/* Display Analysis Result */}
        {result && (
          <div className="mt-4 p-4 border rounded-lg bg-gray-100 text-center">
            <p className="text-xl font-bold">
              Result:{" "}
              <span
                className={result === "Real" ? "text-green-600" : "text-red-600"}
              >
                {result}
              </span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
