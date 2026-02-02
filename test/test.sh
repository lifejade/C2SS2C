source "$(dirname "$0")/config.sh"

echo "환경: $DIR_PATH"
echo "로그 디렉터리: $SIMUL_LOG_PATH"
echo "실행 명령: $TEST_NAME"

ts=$(date '+%y%m%d%H%M')
session="${ts}_${TEST_NAME}"
log="${SIMUL_LOG_PATH}/${ts}_${TEST_NAME}.txt"

cmd="cd \"$DIR_PATH\" && go test -run ^Test_${TEST_NAME}\$ -timeout 100000h > \"$log\""

echo "세션 이름: $session"
echo "로그 파일: $log"
echo "실행할 명령: $cmd"

screen -dmS "$session" bash -c "$cmd"

echo "명령 실행 완료"
