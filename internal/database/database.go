package database

import (
	"errors"
	"fmt"
	"log"
	"os"
)

func InitDatabase() {
	dbType := os.Getenv("DB_TYPE")
	if dbType == "" {
		fmt.Println("DB_TYPE not found, use sqlite as default")
		dbType = "sqlite"
	}
	switch dbType {
	case "sqlite":
		ConnectSQLite()
		break
	default:
		log.Fatalln(errors.New("database type not supported yet"))
	}
}
