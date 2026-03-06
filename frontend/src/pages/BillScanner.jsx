import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileSearch, Activity, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { api } from '../api/client';

export default function BillScanner() {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, scanning, success, error
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const onDrop = useCallback(acceptedFiles => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0]);
            setStatus('idle');
            setResult(null);
            setError('');
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
        maxFiles: 1
    });

    const handleScan = async () => {
        if (!file) return;
        setStatus('scanning');

        try {
            const response = await api.analyseBill(file);
            setResult(response.extracted_data);
            setStatus('success');
        } catch (err) {
            setStatus('error');
            setError(err.response?.data?.detail || err.message || 'Failed to scan bill.');
        }
    };

    return (
        <div className="fade-in" style={{ maxWidth: '900px', margin: '0 auto' }}>
            <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>
                Bill & Document Scanner
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                Extract structured data directly from hospital bills using high-precision vision intelligence
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 1.5fr', gap: '24px', alignItems: 'flex-start' }}>

                {/* Upload Column */}
                <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FileSearch color="var(--accent-blue)" /> Upload Bill Image
                    </h3>

                    <div
                        {...getRootProps()}
                        style={{
                            border: `2px dashed ${isDragActive ? 'var(--accent-purple)' : 'var(--glass-border)'}`,
                            borderRadius: '12px',
                            padding: '32px',
                            textAlign: 'center',
                            background: isDragActive ? 'rgba(139, 92, 246, 0.05)' : 'var(--bg-secondary)',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            marginBottom: '24px',
                            minHeight: '200px',
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center'
                        }}
                    >
                        <input {...getInputProps()} />
                        {file ? (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                                <img
                                    src={URL.createObjectURL(file)}
                                    alt="preview"
                                    style={{ maxHeight: '120px', borderRadius: '8px', objectFit: 'contain' }}
                                />
                                <p style={{ fontWeight: 500, fontSize: '0.9rem', marginTop: '8px' }}>{file.name}</p>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                                <FileText size={40} color="var(--text-muted)" />
                                <div>
                                    <p style={{ fontWeight: 500 }}>Drag & drop image</p>
                                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>JPG, PNG up to 5MB</p>
                                </div>
                            </div>
                        )}
                    </div>

                    <button
                        className="btn-primary"
                        onClick={handleScan}
                        disabled={!file || status === 'scanning'}
                        style={{ width: '100%', justifyContent: 'center', opacity: (!file || status === 'scanning') ? 0.6 : 1 }}
                    >
                        {status === 'scanning' ? <><Loader2 className="animate-spin" size={18} /> Analyzing Vision...</> : 'Scan Document'}
                    </button>

                    {status === 'error' && (
                        <div style={{ marginTop: '16px', padding: '12px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--error)', fontSize: '0.9rem', display: 'flex', gap: '8px' }}>
                            <AlertCircle size={18} style={{ flexShrink: 0 }} /> {error}
                        </div>
                    )}
                </div>

                {/* Results Column */}
                <div className="glass-panel" style={{ padding: '24px', height: '100%', minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity color="var(--accent-purple)" /> Extraction Results
                    </h3>

                    {status === 'idle' && !result && (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                            <FileSearch size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
                            <p>Upload a bill to see extracted data here</p>
                        </div>
                    )}

                    {status === 'scanning' && (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-blue)' }}>
                            <div className="animate-pulse" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'var(--accent-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px' }}>
                                    <Activity size={40} />
                                </div>
                                <p style={{ fontWeight: 500 }}>The vision intelligence system is analyzing the image...</p>
                                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>Extracting hospital, diagnosis, and amount</p>
                            </div>
                        </div>
                    )}

                    {status === 'success' && result && (
                        <div className="fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            <div style={{ padding: '16px', borderRadius: '8px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <CheckCircle2 size={18} /> Structured data extracted successfully
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                                <div style={{ background: 'var(--bg-color)', padding: '16px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '4px' }}>Hospital Name</p>
                                    <p style={{ fontWeight: 600, fontSize: '1.1rem' }}>{result.hospital_name || 'Not Found'}</p>
                                </div>
                                <div style={{ background: 'var(--bg-color)', padding: '16px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '4px' }}>Patient Name</p>
                                    <p style={{ fontWeight: 600, fontSize: '1.1rem' }}>{result.patient_name || 'Not Found'}</p>
                                </div>
                                <div style={{ background: 'var(--bg-color)', padding: '16px', borderRadius: '8px', border: '1px solid var(--glass-border)', gridColumn: 'span 2' }}>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '4px' }}>Total Amount</p>
                                    <p className="gradient-text" style={{ fontWeight: 700, fontSize: '1.5rem' }}>{result.total_amount || 'Not Found'}</p>
                                </div>
                                <div style={{ background: 'var(--bg-color)', padding: '16px', borderRadius: '8px', border: '1px solid var(--glass-border)', gridColumn: 'span 2' }}>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '4px' }}>Diagnosis</p>
                                    <p style={{ fontWeight: 500 }}>{result.diagnosis || 'Not Found'}</p>
                                </div>
                            </div>

                            {result.breakdown && result.breakdown.length > 0 && (
                                <div style={{ marginTop: '16px' }}>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '12px' }}>Line Items</p>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                        {result.breakdown.map((item, idx) => (
                                            <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', background: 'var(--bg-secondary)', borderRadius: '6px', border: '1px solid var(--glass-border)' }}>
                                                <span style={{ color: 'var(--text-secondary)' }}>{item.category}</span>
                                                <span style={{ fontWeight: 500 }}>{item.amount}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Fallback for raw text if parsing failed but model returned text */}
                            {result.raw_text && !result.hospital_name && (
                                <div style={{ padding: '16px', background: 'var(--bg-color)', borderRadius: '8px', border: '1px solid var(--glass-border)', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.9rem' }}>
                                    {result.raw_text}
                                </div>
                            )}
                        </div>
                    )}

                </div>
            </div>
        </div>
    );
}
