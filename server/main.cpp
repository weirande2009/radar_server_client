#include "include/Server.hpp"
#include "include/Handler.hpp"

int main()
{
    /* allocate a server */
    Server myserver;
    /* start server */
    myserver.start();

    /* allocate a message handler */
    Handler handler;

    int message_size;
    char buffer[1024];

    /* loop of waiting for message */
    while(true){
        message_size = myserver.receive_data(buffer);
        handler.handle_message(buffer, message_size);
    }

    return 0;
}