import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const api = {
    // System Health
    getStatus: () => apiClient.get('/status').then(res => res.data),

    // Feature 11: Policy Management
    getPolicies: (type = null) =>
        apiClient.get('/policies', { params: { policy_type: type } }).then(res => res.data),

    uploadPolicy: (formData) =>
        apiClient.post('/policies/ingest', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        }).then(res => res.data),

    // Feature 12: Chat & General Query
    queryAssistant: (data) => apiClient.post('/query', data).then(res => res.data),

    // Feature 1: Multi-Policy Compare
    comparePolicies: (data) => apiClient.post('/compare', data).then(res => res.data),

    // Feature 14: Claim Checklist
    getClaimChecklist: (data) => apiClient.post('/claim-checklist', data).then(res => res.data),

    // Feature 15/Feature 3: Exclusions
    getExclusions: (data) => apiClient.post('/exclusions', data).then(res => res.data),
    checkExcluded: (data) => apiClient.post('/is-excluded', data).then(res => res.data),

    // Feature 13: Bill Analysis
    analyseBill: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return apiClient.post('/analyse-bill', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        }).then(res => res.data);
    }
};
