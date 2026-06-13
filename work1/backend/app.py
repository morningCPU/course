import os
import redis
from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', 'redis-svc')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

try:
    if REDIS_PASSWORD:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
    else:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f"Warning: Could not connect to Redis: {e}")
    r = None

@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'message': 'Backend is running'
    })

@app.route('/api/data', methods=['GET', 'POST'])
def data():
    if r is None:
        return jsonify({'error': 'Redis not connected'}), 500
    
    if request.method == 'POST':
        key = request.json.get('key', 'default')
        value = request.json.get('value', '')
        r.set(key, value)
        return jsonify({'status': 'saved', 'key': key, 'value': value})
    else:
        keys = r.keys('*')
        data = {k: r.get(k) for k in keys}
        return jsonify(data)

@app.route('/api/health', methods=['GET'])
def health():
    redis_status = 'connected' if r and r.ping() else 'disconnected'
    return jsonify({
        'status': 'healthy',
        'redis': redis_status,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)