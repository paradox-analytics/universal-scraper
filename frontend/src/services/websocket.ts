import { io, Socket } from 'socket.io-client';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8080';

let socket: Socket | null = null;

export const connectWebSocket = (): Socket => {
  if (!socket) {
    socket = io(WS_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    });

    socket.on('connect', () => {
      console.log('WebSocket connected');
    });

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  return socket;
};

export const disconnectWebSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
};

export const subscribeToJob = (jobId: string, callback: (data: any) => void) => {
  const ws = connectWebSocket();
  ws.emit('subscribe', { jobId });
  ws.on(`job:${jobId}`, callback);
  
  return () => {
    ws.off(`job:${jobId}`, callback);
    ws.emit('unsubscribe', { jobId });
  };
};

export const subscribeToCache = (url: string, callback: (data: any) => void) => {
  const ws = connectWebSocket();
  ws.emit('subscribe_cache', { url });
  ws.on(`cache:${url}`, callback);
  
  return () => {
    ws.off(`cache:${url}`, callback);
    ws.emit('unsubscribe_cache', { url });
  };
};

