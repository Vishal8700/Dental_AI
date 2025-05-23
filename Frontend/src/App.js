import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  FileImage, 
  Loader2, 
  CheckCircle, 
  AlertCircle, 
  Download, 
  Zap, 
  BarChart3, 
  Eye, 
  Activity, 
  Calendar,
  TrendingUp,
  Brain,
  Shield,
  AlertTriangle,
  Clock,
  Heart,
  Sun,
  Moon,
  X,
  Target,
  Users,
  Award
} from 'lucide-react';
import './App.css';

const App = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [activeTab, setActiveTab] = useState('annotated');
  const [theme, setTheme] = useState(() => {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  // Theme toggle effect
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e) => {
      setTheme(e.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);
    document.documentElement.setAttribute('data-theme', theme);

    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);
  const [notifications, setNotifications] = useState([]);
  const fileInputRef = useRef(null);

  // Theme toggle effect
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const addNotification = (type, title, message) => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, type, title, message }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    const allowedTypes = ['.dcm', '.rvg', '.png', '.jpg', '.jpeg'];
    const fileExtension = '.' + selectedFile.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      setError('Please select a valid file (.dcm, .rvg, .png, .jpg, .jpeg)');
      addNotification('error', 'Invalid File Type', 'Please select a supported file format');
      return;
    }
    
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      addNotification('error', 'File Too Large', 'Please select a file smaller than 10MB');
      return;
    }
    
    setFile(selectedFile);
    setError('');
    setResults(null);
    addNotification('success', 'File Selected', 'Ready for analysis');
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first');
      addNotification('warning', 'No File Selected', 'Please select a file to analyze');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setResults(data);
        addNotification('success', 'Analysis Complete', `Found ${data.detections.length} pathological findings`);
      } else {
        setError(data.message || 'Analysis failed');
        addNotification('error', 'Analysis Failed', data.message || 'Unknown error occurred');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Failed to connect to the server. Please ensure the API is running on port 8000.');
        addNotification('error', 'Connection Error', 'Unable to connect to analysis server');
      } else {
        setError(`Analysis failed: ${err.message}`);
        addNotification('error', 'Analysis Error', err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResults(null);
    setError('');
    setActiveTab('annotated');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    addNotification('success', 'Reset Complete', 'Ready for new analysis');
  };

  const downloadReport = () => {
    if (!results) return;
    
    const reportContent = `DENTAL X-RAY ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}
Processing Time: ${results.processing_time}

ANALYSIS SUMMARY:
- Total Detections: ${results.summary?.total_detections || 0}
- Pathology Types: ${results.summary?.pathology_types || 0}
- High Confidence Detections: ${results.summary?.high_confidence_detections || 0}
- Medium Confidence Detections: ${results.summary?.medium_confidence_detections || 0}
- Low Confidence Detections: ${results.summary?.low_confidence_detections || 0}
- Average Confidence: ${results.summary?.average_confidence ? (results.summary.average_confidence * 100).toFixed(1) + '%' : 'N/A'}

PATHOLOGY DISTRIBUTION:
${results.summary?.pathology_distribution ? 
  Object.entries(results.summary.pathology_distribution)
    .map(([pathology, count]) => `- ${pathology}: ${count} occurrence(s)`)
    .join('\n') : 'No pathologies detected'}

DETAILED REPORT:
${results.report}

DETECTED PATHOLOGIES:
${results.detections.length === 0 ? 'None detected' : 
  results.detections.map((d, index) => 
    `${index + 1}. ${d.class_name}: ${(d.confidence * 100).toFixed(1)}% confidence${d.severity ? ` (${d.severity} severity)` : ''}${d.location_description ? `\n   Location: ${d.location_description}` : ''}`
  ).join('\n')}
`;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dental-analysis-report-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    addNotification('success', 'Report Downloaded', 'Analysis report saved successfully');
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high': return '#dc2626';
      case 'medium': return '#d97706';
      case 'low': return '#059669';
      default: return '#64748b';
    }
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  const calculateConfidencePercentage = (confidence) => {
    return Math.round(confidence * 100);
  };

  return (
    <div className="app">
      {/* Notifications */}
      {notifications.map(notification => (
        <div key={notification.id} className={`notification ${notification.type}`}>
          <div className="notification-header">
            <span className="notification-title">{notification.title}</span>
            <button 
              className="notification-close"
              onClick={() => removeNotification(notification.id)}
            >
              <X size={16} />
            </button>
          </div>
          <div className="notification-message">{notification.message}</div>
        </div>
      ))}

      {/* Header */}
      <div className="header">
        <div className="header-content">
          <div className="logo">
            <Zap className="logo-icon" />
            <h1>DentalAI</h1>
          </div>
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
          </button>
        </div>
      </div>

      <div className="main-content">
        {/* Upload Section */}
        <div className="upload-section">
          <div 
            className={`upload-area ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              accept=".dcm,.rvg,.png,.jpg,.jpeg"
              style={{ display: 'none' }}
            />
            
            {file ? (
              <div className="file-info">
                <FileImage className="file-icon" />
                <div className="file-details">
                  <p className="file-name">{file.name}</p>
                  <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <CheckCircle className="success-icon" />
              </div>
            ) : (
              <div className="upload-placeholder">
                <Upload className="upload-icon" />
                <h3>Drop your X-ray file here</h3>
                <p>or click to browse</p>
                <div className="supported-formats">
                  <span>Supported: DICOM (.dcm), RVG (.rvg), PNG, JPG</span>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle className="error-icon" />
              <span>{error}</span>
            </div>
          )}

          <div className="action-buttons">
            <button 
              className="btn btn-primary" 
              onClick={handleAnalyze}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="btn-icon spinning" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="btn-icon" />
                  Analyze X-Ray
                </>
              )}
            </button>
            
            {(file || results) && (
              <button className="btn btn-secondary" onClick={handleReset}>
                Reset
              </button>
            )}
          </div>
        </div>

        {/* Floating Upload Button */}
        {!file && (
          <button 
            className="floating-upload"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload />
          </button>
        )}

        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2>Analysis Results</h2>
              <div className="results-meta">
                <span className="processing-time">
                  <Activity className="meta-icon" />
                  Processed in {results.processing_time}
                </span>
                <button className="btn btn-outline" onClick={downloadReport}>
                  <Download className="btn-icon" />
                  Download Report
                </button>
              </div>
            </div>

            {/* Diagnostic Dashboard */}
            <div className="diagnostic-dashboard">
              {/* Summary Statistics */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="card-title">
                    <BarChart3 className="card-icon" />
                    <h3>Detection Summary</h3>
                  </div>
                  <span className="card-badge">{results.summary?.total_detections || 0}</span>
                </div>
                
                <div className="summary-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Findings</span>
                    <span className="stat-value">{results.summary?.total_detections || 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Pathology Types</span>
                    <span className="stat-value">{results.summary?.pathology_types || 0}</span>
                  </div>
                </div>

                <div className="progress-container">
                  <div className="progress-label">
                    <span>Analysis Confidence</span>
                    <span>{results.summary?.average_confidence ? (results.summary.average_confidence * 100).toFixed(1) + '%' : '0%'}</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${results.summary?.average_confidence ? results.summary.average_confidence * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Confidence Meter */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="card-title">
                    <Target className="card-icon" />
                    <h3>Confidence Analysis</h3>
                  </div>
                </div>
                
                               <div className="confidence-meter" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
                  <div className="meter-circle" style={{ width: '200px', height: '200px', position: 'relative' }}>
                    <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
                      <circle
                        className="meter-background"
                        cx="50"
                        cy="50"
                        r="30"
                        fill="none"
                        strokeWidth="8"
                        style={{ stroke: 'var(--border-color)', opacity: 0.2 }}
                      />
                      <circle
                        className="meter-progress"
                        cx="50"
                        cy="50"
                        r="30"
                        fill="none"
                        strokeWidth="8"
                        strokeLinecap="round"
                        style={{
                          stroke: 'var(--primary-color)',
                          transition: 'stroke-dasharray 0.8s ease-in-out'
                        }}
                        strokeDasharray={`${(results.summary?.average_confidence || 0) * 219.8} 219.8`}
                      />
                    </svg>
                    <div className="meter-text" style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>
                        {results.summary?.average_confidence ? (results.summary.average_confidence * 100).toFixed(0) + '%' : '0%'}
                      </div>
                    </div>
                  </div>
                  <div className="meter-info" style={{ textAlign: 'center' }}>
                    <h4 style={{ fontSize: '1.4rem', marginBottom: '0.75rem', color: 'var(--text-color)' }}>Overall Confidence</h4>
                    <p style={{ fontSize: '1rem', opacity: 0.8, maxWidth: '280px', margin: '0 auto' }}>Average detection confidence across all findings</p>
                  </div>
                </div>
              </div>

              {/* Risk Assessment */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="card-title">
                    <Shield className="card-icon" />
                    <h3>Risk Assessment</h3>
                  </div>
                </div>
                
                <div className="risk-assessment">
                  <div className="risk-item">
                    <div className="risk-info">
                      <h5>High Priority</h5>
                      <p>Requires immediate attention</p>
                    </div>
                    <div className="risk-level risk-high">
                      <div className="risk-dot"></div>
                      {results.summary?.high_confidence_detections || 0}
                    </div>
                  </div>
                  
                  <div className="risk-item">
                    <div className="risk-info">
                      <h5>Monitor</h5>
                      <p>Regular observation needed</p>
                    </div>
                    <div className="risk-level risk-medium">
                      <div className="risk-dot"></div>
                      {results.summary?.medium_confidence_detections || 0}
                    </div>
                  </div>
                  
                  <div className="risk-item">
                    <div className="risk-info">
                      <h5>Routine Care</h5>
                      <p>Standard preventive measures</p>
                    </div>
                    <div className="risk-level risk-low">
                      <div className="risk-dot"></div>
                      {results.summary?.low_confidence_detections || 0}
                    </div>
                  </div>
                </div>
              </div>

              {/* Treatment Recommendations */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="card-title">
                    <Heart className="card-icon" />
                    <h3>Treatment Priority</h3>
                  </div>
                </div>
                
                <div className="treatment-recommendations">
                  {results.detections.slice(0, 3).map((detection, index) => (
                    <div key={index} className="treatment-item">
                      <div className="treatment-header">
                        <span className={`treatment-priority priority-${detection.severity?.toLowerCase() || 'medium'}`}>
                          {detection.severity || 'Medium'}
                        </span>
                        <span className="treatment-title">{detection.class_name}</span>
                      </div>
                      <div className="treatment-description">
                        Confidence: {(detection.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                  
                  {results.detections.length === 0 && (
                    <div className="treatment-item">
                      <div className="treatment-header">
                        <span className="treatment-priority priority-low">Routine</span>
                        <span className="treatment-title">Preventive Care</span>
                      </div>
                      <div className="treatment-description">
                        Continue regular dental hygiene and checkups
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Pathology Distribution */}
            {results.summary?.pathology_distribution && Object.keys(results.summary.pathology_distribution).length > 0 && (
              <div className="dashboard-card" style={{gridColumn: '1 / -1'}}>
                <div className="card-header">
                  <div className="card-title">
                    <Brain className="card-icon" />
                    <h3>Pathology Distribution</h3>
                  </div>
                </div>
                
                <div className="pathology-distribution">
                  {Object.entries(results.summary.pathology_distribution).map(([pathology, count], index) => (
                    <div key={pathology} className="progress-container">
                      <div className="progress-label">
                        <span style={{textTransform: 'capitalize'}}>{pathology}</span>
                        <span>{count} occurrence{count !== 1 ? 's' : ''}</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill"
                          style={{ 
                            width: `${(count / results.summary.total_detections) * 100}%`,
                            background: `hsl(${index * 60}, 70%, 50%)`
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Image Analysis Section */}
            <div className="results-content">
              <div className="image-panel">
                <div className="image-tabs">
                  <button 
                    className={`tab-btn ${activeTab === 'annotated' ? 'active' : ''}`}
                    onClick={() => setActiveTab('annotated')}
                  >
                    <Eye className="tab-icon" />
                    Annotated View
                  </button>
                  <button 
                    className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`}
                    onClick={() => setActiveTab('original')}
                  >
                    <FileImage className="tab-icon" />
                    Original View
                  </button>
                </div>

                <div className="image-container">
                  <img 
                    src={`data:image/png;base64,${activeTab === 'annotated' ? results.annotated_image : results.original_image}`}
                    alt={activeTab === 'annotated' ? 'Annotated X-ray' : 'Original X-ray'}
                    className="xray-image"
                  />
                  
                  {results.detections.length > 0 && activeTab === 'annotated' && (
                    <div className="detections-overlay">
                      <div className="detections-count">
                        {results.detections.length} patholog{results.detections.length === 1 ? 'y' : 'ies'} detected
                      </div>
                    </div>
                  )}
                </div>

                {/* Detailed Detections List */}
                {results.detections.length > 0 && (
                  <div className="detections-list">
                    <h4>Detected Pathologies</h4>
                    
                    {results.detections.map((detection, index) => (
                      <div key={index} className="detection-item">
                        <div className="detection-header">
                          <span className="detection-id">#{index + 1}</span>
                          <span className="detection-name">{detection.class_name}</span>
                          {detection.severity && (
                            <span 
                              className="severity-badge"
                              style={{ backgroundColor: getSeverityColor(detection.severity) }}
                            >
                              {detection.severity}
                            </span>
                          )}
                        </div>
                        
                        <div className="detection-details">
                          <div className="confidence-info">
                            <span className="confidence-label">Confidence:</span>
                            <span className="confidence-value">
                              {(detection.confidence * 100).toFixed(1)}%
                            </span>
                            <span className="confidence-level">
                              ({getConfidenceLevel(detection.confidence)})
                            </span>
                          </div>
                          
                          {detection.location_description && (
                            <div className="location-info">
                              <span className="location-label">Location:</span>
                              <span className="location-value">{detection.location_description}</span>
                            </div>
                          )}
                        </div>
                        
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill"
                            style={{ 
                              width: `${detection.confidence * 100}%`,
                              backgroundColor: getSeverityColor(detection.severity)
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Diagnostic Report Panel */}
              <div className="report-panel">
                <div className="report-header">
                  <h3>Diagnostic Report</h3>
                  {results.summary && (
                    <div className="report-stats">
                      <span className="report-stat">
                        {results.summary.total_detections} findings
                      </span>
                      <span className="report-stat">
                        {results.summary.pathology_types} types
                      </span>
                    </div>
                  )}
                </div>
                
                <div className="report-content">
                  <pre className="report-text">{results.report}</pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="footer">
        <p>&copy; 2025 DentalAI. Advanced dental diagnostics powered by AI.</p>
        <p>Made with <Heart size={16} className="heart-icon" /> by <a href="https://github.com/Vishal8700" target="_blank" rel="noopener noreferrer">gitalien</a></p>
      </div>
    </div>
  );
};

export default App;
