#ifndef INCLUDE_SERVER_H
#define INCLUDE_SERVER_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<iostream>


class Server
{
protected:
    /* socket number */
    int socket_fd;
    /* client number */
    int client_fd;
    /* server address */
    struct sockaddr_in addr;
    /* client address */
    struct sockaddr_in client;

    /* host and port*/
    const char host[10] = "127.0.0.1";
    const int port = 8888;


public:
    Server(void)
    {
        //1.创建一个socket
        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1){
            std::cout << "socket fail： " << std::endl;
            exit(1);
        }   
        //2.准备通讯地址（必须是服务器的）192.168.1.49是本机的IP
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);//将一个无符号短整型的主机数值转换为网络字节顺序，即大尾顺序(big-endian)
        addr.sin_addr.s_addr = inet_addr(host);//net_addr方法可以转化字符串，主要用来将一个十进制的数转化为二进制的数，用途多于ipv4的IP转化。
        //3.bind()绑定
        //参数一：socket_fd
        //参数二：(struct sockaddr*)&addr 前面结构体，即地址
        //参数三: addr结构体的长度
        int res = bind(socket_fd, (struct sockaddr*)&addr, sizeof(addr));
        if (res == -1)
        {
            std::cout << "bind fail： " << std::endl;
            exit(-1);
        }
        std::cout << "bind ok! Wait for connecting" << std::endl;
        //4.监听客户端listen()函数
        //参数二：进程上限，一般小于30
        listen(socket_fd, 30);
    }
    ~Server(void){}
    /* start to waiting for connecting */
    void start()
    {
        //等待客户端的连接accept()，返回用于交互的socket描述符
        socklen_t len = sizeof(client);
        std::cout << "listening......" << std::endl;
        client_fd = accept(socket_fd, (struct sockaddr*)&client, &len);
        if (client_fd == -1)
        {
            std::cout << "accept fail\n" << std::endl;
            exit(-1);
        }
        write(client_fd, "welcome", 7);
    }

    /* waiting for message from client */
    /* received data will be in buffer */
    /* return the size of received data */
    int receive_data(char buffer[1024])
    {
        //第一个参数：accept 返回的文件描述符
        //第二个参数：存放读取的内容
        //第三个参数：内容的大小
        int size = read(client_fd, buffer, sizeof(buffer));//通过fd与客户端联系在一起,返回接收到的字节数
        std::cout << "received bytes length: " << size << std::endl;
        std::cout << "content:  " << buffer << std::endl;
        return size;
    }
};

#endif /*INCLUDE_TEST_SERVER_H*/