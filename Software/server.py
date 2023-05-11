import socket

# 替换为你的PC端IP地址和端口
server_ip = 'your_PC_IP'
server_port = 12345

# 创建一个socket对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP地址和端口
server_socket.bind((server_ip, server_port))

# 开始监听连接
server_socket.listen(1)
print(f"Server is listening on {server_ip}:{server_port}")

while True:
    # 接受客户端连接
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    while True:
        try:
            # 接收数据
            data = client_socket.recv(1024)
            if not data:
                break

            # 打印接收到的数据
            print(data.decode('utf-8'))

        except ConnectionResetError:
            print("Client disconnected")
            break

    # 关闭客户端连接
    client_socket.close()

# 关闭服务器socket（通常不会执行到这里）
server_socket.close()
