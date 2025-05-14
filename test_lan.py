import socket
import asyncio
import websockets

# ─────────────────────────────────────────────
# 1. Gửi UDP broadcast để tìm server
def discover_server():
    BROADCAST_PORT = 8888
    MESSAGE = "123"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(5)

    sock.sendto(MESSAGE.encode(), ("255.255.255.255", BROADCAST_PORT))
    print("📤 Broadcast sent...")

    try:
        data, addr = sock.recvfrom(1024)
        response = data.decode()
        if response.startswith("ACK|"):
            server_ip = response.split("|")[1]
            print(f"✅ Server found at {server_ip}")
            return server_ip
    except socket.timeout:
        print("❌ No response from server.")
        return None

# ─────────────────────────────────────────────
# 2. Kết nối WebSocket đến server
async def connect_ws(ip):
    uri = f"ws://{ip}:9000/ws"
    async with websockets.connect(uri) as websocket:
        print("🔗 Connected to WebSocket")
        await websocket.send("Hello from Pi")
        ack = await websocket.recv()
        print("📥 Received from server:", ack)

# ─────────────────────────────────────────────
# 3. Tổng hợp
def main():
    ip = discover_server()
    if ip:
        asyncio.run(connect_ws(ip))

if __name__ == "__main__":
    main()
