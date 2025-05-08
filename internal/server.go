package internal

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

func init() {
	r := gin.Default()
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})
	err := r.Run() // listen and serve on 0.0.0.0:8080 (for windows "localhost:8080")
	if err != nil {
		fmt.Println("Failed to start server:", err)
		return
	}
	fmt.Println("Server started on port 8080")
}
