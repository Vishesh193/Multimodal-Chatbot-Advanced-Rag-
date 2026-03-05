import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { api } from '../api/client';

export default function PolicyUpload() {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, uploading, success, error
    const [message, setMessage] = useState('');
    const [formData, setFormData] = useState({
        policy_name: '',
        insurer: '',
        policy_type: 'health',
        policy_number: '',
        sum_insured: '',
        premium: '',
        holder_name: '',
        tags_string: ''
    });

    const onDrop = useCallback(acceptedFiles => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0]);
            setStatus('idle');
            setMessage('');
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        maxFiles: 1
    });

    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) {
            setStatus('error');
            setMessage('Please select a PDF file to upload.');
            return;
        }
        if (!formData.policy_name || !formData.insurer) {
            setStatus('error');
            setMessage('Policy Name and Insurer are required.');
            return;
        }

        setStatus('uploading');

        try {
            const uploadData = new FormData();
            uploadData.append('file', file);
            Object.keys(formData).forEach(key => {
                uploadData.append(key, formData[key]);
            });

            const response = await api.uploadPolicy(uploadData);
            setStatus('success');
            setMessage(`Successfully ingested! Policy ID: ${response.policy_id}`);
            setFile(null);
            // Reset form but keep type
            setFormData({
                policy_name: '', insurer: '', policy_type: 'health', policy_number: '',
                sum_insured: '', premium: '', holder_name: '', tags_string: ''
            });
        } catch (err) {
            setStatus('error');
            setMessage(err.response?.data?.detail || err.message || 'Failed to upload policy.');
        }
    };

    return (
        <div className="fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
            <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>
                Policy Ingestion
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                Add new insurance documents into the RAG vector store for analysis.
            </p>

            <form onSubmit={handleUpload} className="glass-panel" style={{ padding: '32px' }}>
                <div
                    {...getRootProps()}
                    style={{
                        border: `2px dashed ${isDragActive ? 'var(--accent-blue)' : 'var(--glass-border)'}`,
                        borderRadius: '12px',
                        padding: '40px',
                        textAlign: 'center',
                        background: isDragActive ? 'rgba(59, 130, 246, 0.05)' : 'var(--bg-secondary)',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        marginBottom: '24px'
                    }}
                >
                    <input {...getInputProps()} />
                    {file ? (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                            <FileText size={48} color="var(--accent-blue)" />
                            <p style={{ fontWeight: 500 }}>{file.name}</p>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                            <UploadCloud size={48} color="var(--text-muted)" />
                            <div>
                                <p style={{ fontWeight: 500, fontSize: '1.1rem' }}>Drag & drop policy PDF here</p>
                                <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>or click to browse files</p>
                            </div>
                        </div>
                    )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
                    <div>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Policy Name *</label>
                        <input
                            name="policy_name"
                            value={formData.policy_name}
                            onChange={handleInputChange}
                            placeholder="e.g. Family Floater Base"
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-color)', color: 'var(--text-primary)' }}
                        />
                    </div>
                    <div>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Insurer *</label>
                        <input
                            name="insurer"
                            value={formData.insurer}
                            onChange={handleInputChange}
                            placeholder="e.g. Star Health"
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-color)', color: 'var(--text-primary)' }}
                        />
                    </div>
                    <div>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Policy Type</label>
                        <select
                            name="policy_type"
                            value={formData.policy_type}
                            onChange={handleInputChange}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-color)', color: 'var(--text-primary)' }}
                        >
                            <option value="health">Health Insurance</option>
                            <option value="motor">Motor Insurance</option>
                            <option value="life">Life Insurance</option>
                            <option value="travel">Travel Insurance</option>
                            <option value="home">Home Insurance</option>
                        </select>
                    </div>
                    <div>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Policy Number</label>
                        <input
                            name="policy_number"
                            value={formData.policy_number}
                            onChange={handleInputChange}
                            placeholder="e.g. P/123456/01"
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-color)', color: 'var(--text-primary)' }}
                        />
                    </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '32px' }}>
                    <div style={{ flex: 1 }}>
                        {status === 'error' && <p style={{ color: 'var(--error)', display: 'flex', alignItems: 'center', gap: '8px' }}><AlertCircle size={18} /> {message}</p>}
                        {status === 'success' && <p style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle2 size={18} /> {message}</p>}
                    </div>

                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={status === 'uploading'}
                        style={{ opacity: status === 'uploading' ? 0.7 : 1 }}
                    >
                        {status === 'uploading' ? <><Loader2 className="animate-spin" size={18} /> Ingesting to Vector Database...</> : 'Ingest Policy'}
                    </button>
                </div>
            </form>
        </div>
    );
}
