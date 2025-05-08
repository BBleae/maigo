package internal

import (
	"fmt"
	"github.com/joho/godotenv"
)

type GlobalConfig struct {
	Host string
	Port string
}

func LoadEnv() {
	err := godotenv.Load()
	if err != nil {
		fmt.Println(".env file was not loaded")
	}
	fmt.Println(".env file was loaded")
}
